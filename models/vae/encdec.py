import torch
import torch.nn as nn

from models.skeleton.conv import ResSTConv
from models.skeleton.pool import STPool, STUnpool, adj_list_to_edges
from models.skeleton.conv import get_activation


from utils.paramUtil import kit_adj_list, t2m_adj_list, hhi_adj_list

class MotionEncoder(nn.Module):
    def __init__(self, opt):
        super(MotionEncoder, self).__init__()

        self.pose_dim = opt.pose_dim
        self.joints_num = (self.pose_dim + 1) // 12
        if opt.dataset_name == "interhuman" or opt.dataset_name == "interx":
            self.joints_num = opt.joints_num
        self.latent_dim = opt.latent_dim
        self.contact_joints = opt.contact_joints

        self.dataset_name = opt.dataset_name

        self.layers = nn.ModuleList()
        for i in range(self.joints_num):
            if i == 0 and opt.dataset_name == "interx":
                input_dim = 12
            elif i == 0 and opt.dataset_name != "interhuman":
                input_dim = 7
            elif self.dataset_name == "interx":
                input_dim = 12
            elif i in self.contact_joints:
                input_dim = 13
            else:
                input_dim = 12
            self.layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, self.latent_dim),
                    get_activation(opt.activation),
                    nn.Linear(self.latent_dim, self.latent_dim),
                )
            )

    def preprocess(self, x):
        B, T = x.shape[:2]

        if self.dataset_name == "interhuman":
            pos = x[..., : self.joints_num * 3].reshape(
                [x.shape[0], x.shape[1], self.joints_num, 3]
            )
            vel = x[..., self.joints_num * 3 : self.joints_num * 3 * 2].reshape(
                [x.shape[0], x.shape[1], self.joints_num, 3]
            )

            rot = x[
                ...,
                self.joints_num * 3 * 2 : self.joints_num * 3 * 2
                + (self.joints_num - 1) * 6,
            ].reshape([x.shape[0], x.shape[1], self.joints_num - 1, 6])
            rot = torch.cat(
                [torch.zeros(rot.shape[0], rot.shape[1], 1, 6).to(x.device), rot], dim=2
            )

            contact = x[..., -4:]
            joints = []
            # j * [B, T, 12 or 13]
            for i in range(0, self.joints_num):
                joints.append(
                    torch.cat([pos[:, :, i], rot[:, :, i], vel[:, :, i]], dim=-1)
                )
            for cidx, jidx in enumerate(self.contact_joints):
                joints[jidx] = torch.cat(
                    [joints[jidx], contact[:, :, cidx, None]], dim=-1
                )
        elif self.dataset_name == "interx":
            # import ipdb; ipdb.set_trace()
            joints = [
                x[:,:, i, :] for i in range(self.joints_num)
            ]
        else:
            # split
            root, ric, rot, vel, contact = torch.split(
                x,
                [
                    4,
                    3 * (self.joints_num - 1),
                    6 * (self.joints_num - 1),
                    3 * self.joints_num,
                    4,
                ],
                dim=-1,
            )
            ric = ric.reshape(B, T, self.joints_num - 1, 3)
            rot = rot.reshape(B, T, self.joints_num - 1, 6)
            vel = vel.reshape(B, T, self.joints_num, 3)

            # joint-wise input
            joints = [torch.cat([root, vel[:, :, 0]], dim=-1)]  # [B, T, 7]]

            for i in range(1, self.joints_num):
                joints.append(
                    torch.cat(
                        [ric[:, :, i - 1], rot[:, :, i - 1], vel[:, :, i]], dim=-1
                    )
                )
            for cidx, jidx in enumerate(self.contact_joints):
                joints[jidx] = torch.cat(
                    [joints[jidx], contact[:, :, cidx, None]], dim=-1
                )

        return joints

    def forward(self, x):
        """
        x: [bs, nframes, pose_dim]

        nfeats = 12J + 1
            - root_rot_velocity (B, seq_len, 1)
            - root_linear_velocity (B, seq_len, 2)
            - root_y (B, seq_len, 1)
            - ric_data (B, seq_len, (joint_num - 1)*3)
            - rot_data (B, seq_len, (joint_num - 1)*6)
            - local_velocity (B, seq_len, joint_num*3)
            - foot contact (B, seq_len, 4)
        """

        joints = self.preprocess(x=x)

        # encode
        out = []
        for i in range(self.joints_num):
            out.append(self.layers[i](joints[i]))
        out = torch.stack(out, dim=2)

        return out


class MotionDecoder(nn.Module):
    def __init__(self, opt):
        super(MotionDecoder, self).__init__()

        self.dataset_name = opt.dataset_name

        self.pose_dim = opt.pose_dim
        self.joints_num = (self.pose_dim + 1) // 12
        if opt.dataset_name == "interhuman" or opt.dataset_name == "interx":
            self.joints_num = opt.joints_num
        self.latent_dim = opt.latent_dim
        self.contact_joints = opt.contact_joints

        # network components
        self.layers = nn.ModuleList()
        for i in range(self.joints_num):
            if i == 0 and self.dataset_name == "interx":
                output_dim = 12
            elif i == 0 and self.dataset_name != "interhuman":
                output_dim = 7
            elif self.dataset_name == "interx":
                output_dim = 12
            elif i in self.contact_joints:
                output_dim = 13
            else:
                output_dim = 12
            self.layers.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.latent_dim),
                    get_activation(opt.activation),
                    nn.Linear(self.latent_dim, output_dim),
                )
            )

    def postprocess(self, out):
        J = len(out)
        B, T = out[0].shape[:2]

        motion = None

        if self.dataset_name == "interhuman":
            ric_list, rot_list, vel_list = [], [], []
            for i in range(0, self.joints_num):
                ric = out[i][:, :, :3]
                rot = out[i][:, :, 3:9]
                vel = out[i][:, :, 9:12]

                ric_list.append(ric)
                rot_list.append(rot)
                vel_list.append(vel)

            contact = [out[i][:, :, -1] for i in self.contact_joints]
            ric = torch.stack(ric_list, dim=2).reshape(B, T, J * 3)
            rot = torch.stack(rot_list, dim=2).reshape(B, T, J * 6)
            vel = torch.stack(vel_list, dim=2).reshape(B, T, J * 3)
            contact = torch.stack(contact, dim=2).reshape(
                B, T, len(self.contact_joints)
            )

            motion = torch.cat(
                [
                    ric,  # pos
                    vel,  # vel
                    rot[..., 6:],  # rot, expect root
                    contact,  # contact
                ],
                dim=-1,
            )
        elif self.dataset_name == "interx":
            motion = torch.stack(out, dim=2)
        else:
            root = out[0]
            ric_list, rot_list, vel_list = [], [], []
            for i in range(1, self.joints_num):
                ric = out[i][:, :, :3]
                rot = out[i][:, :, 3:9]
                vel = out[i][:, :, 9:12]

                ric_list.append(ric)
                rot_list.append(rot)
                vel_list.append(vel)

            contact = [out[i][:, :, -1] for i in self.contact_joints]

            ric = torch.stack(ric_list, dim=2).reshape(B, T, (J - 1) * 3)
            rot = torch.stack(rot_list, dim=2).reshape(B, T, (J - 1) * 6)
            vel = torch.stack(vel_list, dim=2).reshape(B, T, (J - 1) * 3)
            contact = torch.stack(contact, dim=2).reshape(
                B, T, len(self.contact_joints)
            )

            motion = torch.cat(
                [
                    root[..., :4],  # root
                    ric,  # ric
                    rot,  # rot
                    torch.cat([root[..., 4:], vel], dim=-1),  # vel
                    contact,  # contact
                ],
                dim=-1,
            )
        return motion

    def forward(self, x):
        """
        x: [bs, nframes, joints_num, latent_dim]
        """

        out = []
        for i in range(self.joints_num):
            out.append(self.layers[i](x[:, :, i]))

        motion = self.postprocess(out)

        return motion


class STConvEncoder(nn.Module):
    def __init__(self, opt):
        super(STConvEncoder, self).__init__()

        # adjacency list
        self.adj_list = {
            "t2m": t2m_adj_list,
            "interhuman": t2m_adj_list,
            "interx": hhi_adj_list,
            "kit": kit_adj_list,
        }[opt.dataset_name]

        # topology
        self.edge_list = [adj_list_to_edges(self.adj_list)]
        self.mapping_list = []

        # network
        self.layers = nn.ModuleList()
        for i in range(opt.n_layers):
            layers = []
            for _ in range(opt.n_extra_layers):
                layers.append(
                    ResSTConv(
                        self.edge_list[-1],
                        opt.latent_dim,
                        opt.kernel_size,
                        activation=opt.activation,
                        norm=opt.norm,
                        dropout=opt.dropout,
                    )
                )
            layers.append(
                ResSTConv(
                    self.edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout,
                )
            )

            pool = STPool(opt.dataset_name, i)
            layers.append(pool)
            self.layers.append(nn.Sequential(*layers))

            self.edge_list.append(pool.new_edges)
            self.mapping_list.append(pool.skeleton_mapping)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        for seq in self.layers:
            for layer in seq:
                if isinstance(layer, ResSTConv):
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)
        return x


class STConvDecoder(nn.Module):
    def __init__(self, opt, encoder: STConvEncoder):
        super(STConvDecoder, self).__init__()

        # network modules
        self.layers = nn.ModuleList()

        # build network
        mapping_list = encoder.mapping_list.copy()
        edge_list = encoder.edge_list.copy()

        for i in range(opt.n_layers):
            layers = []

            # unpooling
            layers.append(STUnpool(skeleton_mapping=mapping_list.pop()))

            # conv
            edges = edge_list.pop()
            for _ in range(opt.n_extra_layers):
                layers.append(
                    ResSTConv(
                        edge_list[-1],
                        opt.latent_dim,
                        opt.kernel_size,
                        activation=opt.activation,
                        norm=opt.norm,
                        dropout=opt.dropout,
                    )
                )
            layers.append(
                ResSTConv(
                    edge_list[-1],
                    opt.latent_dim,
                    opt.kernel_size,
                    activation=opt.activation,
                    norm=opt.norm,
                    dropout=opt.dropout,
                )
            )

            self.layers.append(nn.Sequential(*layers))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        x: [B, T, J_in, D]
        out: [B, T, J_out, D]
        """
        for seq in self.layers:
            for layer in seq:
                if isinstance(layer, ResSTConv):
                    x = layer(x, feat_cache, feat_idx)
                else:
                    x = layer(x)
        return x

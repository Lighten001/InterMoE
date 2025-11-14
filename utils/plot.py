import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from utils import paramUtil


from os.path import join as pjoin


def plot_interaction(motions, caption, vis_dir, file_name):
    """
    motions: (seq_len, 2, 262)
    """
    sequences = []
    sequences_ik = []
    foots = []
    for j in range(2):
        motion = motions[:, j]
        joints3d = motion[:, : 22 * 3].reshape(-1, 22, 3)
        joints3d = gaussian_filter1d(joints3d, 1, axis=0, mode="nearest")
        sequences.append(joints3d)

        rot6d = motion[:, 22 * 3 * 2 : -4].reshape(-1, 21, 6)
        rot6d = np.concatenate([np.zeros((rot6d.shape[0], 1, 6)), rot6d], axis=1)

        ik_joint, foot = remove_fs(
            joints3d,
            None,
            fid_l=(7, 10),
            fid_r=(8, 11),
            interp_length=5,
            force_on_floor=True,
        )
        ik_joint = gaussian_filter1d(ik_joint, 3, axis=0, mode="nearest")
        sequences_ik.append(ik_joint)
        foots.append(foot)

    plot_3d_motion_2views(
        pjoin(vis_dir, file_name + ".mp4"),
        paramUtil.t2m_kinematic_chain,
        sequences_ik,
        title=caption,
        fps=30,
        foots=None,
    )


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(
    save_path, kinematic_tree, mp_joints, title, figsize=(10, 10), fps=120, radius=4
):
    matplotlib.use("Agg")

    title_sp = title.split(" ")
    if len(title_sp) > 20:
        title = "\n".join(
            [
                " ".join(title_sp[:10]),
                " ".join(title_sp[10:20]),
                " ".join(title_sp[20:]),
            ]
        )
    elif len(title_sp) > 10:
        title = "\n".join([" ".join(title_sp[:10]), " ".join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz],
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()

    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])
    # print(frame_number)

    # colors = ['red', 'blue', 'black', 'red', 'blue',
    #           'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
    #           'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    #
    colors = [
        "red",
        "green",
        "black",
        "red",
        "blue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkred",
        "darkred",
        "darkred",
        "darkred",
        "darkred",
    ]

    mp_offset = list(range(-len(mp_joints) // 2, len(mp_joints) // 2, 1))
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    for i, joints in enumerate(mp_joints):

        # (seq_len, joints_num, 3)
        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)

        #     print(data.shape)

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        # data[:, :, 0] -= data[0:1, 0:1, 0]
        # data[:, :, 0] += mp_offset[i]
        #
        # data[:, :, 2] -= data[0:1, 0:1, 2]
        mp_data.append(
            {
                "joints": data,
                "MINS": MINS,
                "MAXS": MAXS,
                "trajec": trajec,
            }
        )

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 15  # 7.5
        #         ax =
        plot_xzPlane(-3, 3, 0, -3, 3)
        for pid, data in enumerate(mp_data):
            for i, (chain, color) in enumerate(zip(kinematic_tree, mp_colors[pid])):
                #             print(color)
                if i < 5:
                    linewidth = 2.0
                else:
                    linewidth = 1.0
                ax.plot3D(
                    data["joints"][index, chain, 0],
                    data["joints"][index, chain, 1],
                    data["joints"][index, chain, 2],
                    linewidth=linewidth,
                    color=color,
                )
        #         print(trajec[:index, 0].shape)

        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(
        fig, update, frames=frame_number, interval=1000 / fps, repeat=False
    )

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()


def plot_3d_motion_2views(
    save_path,
    kinematic_tree,
    mp_joints,
    title,
    figsize=(20, 10),
    fps=120,
    radius=8,
    foots=None,
):
    matplotlib.use("Agg")

    title_sp = title.split(" ")
    if len(title_sp) > 20:
        title = "\n".join(
            [
                " ".join(title_sp[:10]),
                " ".join(title_sp[10:20]),
                " ".join(title_sp[20:]),
            ]
        )
    elif len(title_sp) > 10:
        title = "\n".join([" ".join(title_sp[:10]), " ".join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 4])
        ax.set_zlim3d([0, radius / 4])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz],
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    fig = plt.figure(figsize=figsize)
    axs = []
    axs.append(fig.add_subplot(1, 2, 1, projection="3d"))
    axs.append(fig.add_subplot(1, 2, 2, projection="3d"))
    for ax in axs:
        init()

    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])
    colors = [
        "orange",
        "green",
        "black",
        "red",
        "blue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkred",
        "darkred",
        "darkred",
        "darkred",
        "darkred",
    ]

    mp_offset = list(range(-len(mp_joints) // 2, len(mp_joints) // 2, 1))
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    rot = R.from_euler("y", 110, degrees=True)
    for i, joints in enumerate(mp_joints):

        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)

        height_offset = MINS[1]
        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]

        data_rot = rot.apply(data.reshape(-1, 3)).reshape(-1, 22, 3)

        mp_data.append(
            {
                "joints": data,
                "joints_rot": data_rot,
                "MINS": MINS,
                "MAXS": MAXS,
                "trajec": trajec,
            }
        )

    def update(index):
        for ax in axs:
            ax.lines = []
            ax.collections = []

            ax.dist = 15
            plot_xzPlane(ax, -3, 3, 0, -3, 3)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.view_init(elev=120, azim=270)
            ax.axis("off")

        for pid, data in enumerate(mp_data):
            for i, (chain, color) in enumerate(zip(kinematic_tree, mp_colors[pid])):
                if i < 5:
                    linewidth = 2.0
                else:
                    linewidth = 1.0
                if foots is not None:
                    l_heel, l_toe, r_heel, r_toe = foots[pid][:, index]
                    if l_toe == 1:
                        color = "darkred"
                axs[0].plot3D(
                    data["joints"][index, chain, 0],
                    data["joints"][index, chain, 1],
                    data["joints"][index, chain, 2],
                    linewidth=linewidth,
                    color=color,
                )
                axs[1].plot3D(
                    data["joints_rot"][index, chain, 0],
                    data["joints_rot"][index, chain, 1],
                    data["joints_rot"][index, chain, 2],
                    linewidth=linewidth,
                    color=color,
                )

    ani = FuncAnimation(
        fig, update, frames=frame_number, interval=1000 / fps, repeat=False
    )

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()


def remove_fs(
    glb, foot_contact, fid_l=(3, 4), fid_r=(7, 8), interp_length=5, force_on_floor=True
):
    # glb_height = 2.06820832 Not the case, may be use upper leg length
    scale = 1.0  # glb_height / 1.65 #scale to meter
    # fps = 20 #
    # velocity_thres = 10. # m/s
    height_thres = [0.12, 0.05]  # [0.06, 0.03] #[ankle, toe] meter
    if foot_contact is None:

        def foot_detect(positions, velfactor, heightfactor):
            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            feet_l_h = positions[:-1, fid_l, 1]
            feet_l = (
                ((feet_l_x + feet_l_y + feet_l_z) < velfactor)
                & (feet_l_h < heightfactor)
            ).astype(float)

            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            feet_r_h = positions[:-1, fid_r, 1]

            feet_r = (
                ((feet_r_x + feet_r_y + feet_r_z) < velfactor)
                & (feet_r_h < heightfactor)
            ).astype(float)

            return feet_l, feet_r

        # feet_thre = 0.002
        # feet_vel_thre = np.array([velocity_thres**2, velocity_thres**2]) * scale**2 / fps**2
        feet_vel_thre = np.array([0.001, 0.001])  # np.array([0.05, 0.2])
        # height_thre = np.array([0.06, 0.04]) * scale
        feet_h_thre = np.array(height_thres) * scale
        feet_l, feet_r = foot_detect(
            glb, velfactor=feet_vel_thre, heightfactor=feet_h_thre
        )
        foot = np.concatenate([feet_l, feet_r], axis=-1).transpose(1, 0)  # [4, T-1]
        foot = np.concatenate([foot, foot[:, -1:]], axis=-1)
    else:
        foot = foot_contact.transpose(1, 0)

    T = len(glb)

    fid = list(fid_l) + list(fid_r)
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1], glb[:, fid_r, 1]).min(
        axis=1
    )  # [T, 2] -> [T]
    # print(foot_heights)
    # floor_height = softmin(foot_heights, softness=0.03, axis=0)
    sort_height = np.sort(foot_heights)
    temp_len = len(sort_height)
    floor_height = np.mean(sort_height[int(0.25 * temp_len) : int(0.5 * temp_len)])
    if floor_height > 0.5:  # for motion like swim
        floor_height = 0
    # print(floor_height)
    # floor_height = foot_heights.min()
    # print(floor_height)
    # print(foot)
    # print(foot_heights.min())
    # print(floor_height)
    glb[:, :, 1] -= floor_height
    # anim.positions[:, 0, 1] -= floor_height
    for i, fidx in enumerate(fid):
        fixed = foot[i]  # [T]

        """
        for t in range(T):
            glb[t, fidx][1] = max(glb[t, fidx][1], 0.25)
        """

        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= t - s + 1

            if force_on_floor:
                avg[1] = 0.0

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()

            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(
                    alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                    glb[s, fidx],
                    glb[l, fidx],
                )
                ritp = lerp(
                    alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                    glb[s, fidx],
                    glb[r, fidx],
                )
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)), ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(
                    alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                    glb[s, fidx],
                    glb[l, fidx],
                )
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(
                    alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                    glb[s, fidx],
                    glb[r, fidx],
                )
                glb[s, fidx] = ritp.copy()

    # targetmap = {}
    # for j in range(glb.shape[1]):
    #     targetmap[j] = glb[:, j]

    # ik = BasicInverseKinematics(anim, glb, iterations=5,
    #                             silent=True)

    # slightly larger loss, but better visual
    # ik = JacobianInverseKinematics(anim, targetmap, iterations=30, damping=5, recalculate=False, silent=True)

    # anim = ik()
    return glb, foot


def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r

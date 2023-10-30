import matplotlib.pyplot as plt
import numpy as np

# h36m骨架连接顺序，每个骨架三个维度，分别为：起始关节，终止关节，左右关节标识(1 left 0 right)
human36m_connectivity_dict = [[0, 1, 0], [1, 2, 0], [2, 6, 0], [5, 4, 1], [4, 3, 1], [3, 6, 1], [6, 7, 0],
                              [7, 8, 0], [8, 16, 0], [9, 16, 0],
                              [8, 12, 0], [11, 12, 0], [10, 11, 0], [8, 13, 1], [13, 14, 1], [14, 15, 1]]


def draw3Dpose(pose_3d, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):  # blue, orange
    for i in human36m_connectivity_dict:
        x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if i[2] else rcolor)

    RADIUS = 10  # space around the subject
    xroot, yroot, zroot = pose_3d[5, 0], pose_3d[5, 1], pose_3d[5, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([0, 2 * RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


if __name__ == '__main__':
    
    specific_3d_skeleton = np.load("imp_datasets/video_datasets/data/dance/S3D_l8/A001/A001P001N001S001/S3D.npy")

    # specific_3d_skeleton = np.array([[-60.70762, 403.8008, 109.0549],
    #                                  [-21.792213, 350.7608, 542.72705],
    #                                  [19.21995, 377.59323, 992.2105],
    #                                  [306.85883, 326.98297, 967.81555],
    #                                  [280.36816, 338.44727, 516.5916],
    #                                  [255.34781, 397.92474, 82.73034],
    #                                  [163.04, 352.288, 980.013],
    #                                  [178.85904, 334.66092, 1240.1534],
    #                                  [187.37048, 291.85312, 1487.3542],
    #                                  [150.81487, 211.43636, 1683.0452],
    #                                  [-384.66232, 412.21793, 1211.5632],
    #                                  [-168.92372, 486.90985, 1298.1694],
    #                                  [34.860744, 360.55948, 1461.1226],
    #                                  [353.3462, 285.74405, 1454.4487],
    #                                  [610.2841, 259.90445, 1322.6831],
    #                                  [778.4938, 93.56336, 1262.1969],
    #                                  [164.2095, 206.31268, 1568.9426]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw3Dpose(specific_3d_skeleton, ax)
    plt.show()
    plt.savefig("1.jpg")


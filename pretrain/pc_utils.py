""" Utility functions for processing point clouds.

Author: Charles R. Qi, Hao Su
Date: November 2016
"""
import os
import sys
import warnings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


# Draw point cloud
from eulerangles import euler2mat
import math
# Point cloud IO
import numpy as np
from plyfile import PlyData, PlyElement
import torch
import random
# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b,:,:]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)

def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    return vol

#a = np.zeros((16,1024,3))
#print point_cloud_to_volume_batch(a, 12, 1.0, False).shape

def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] == 1:
                    points.append(np.array([a,b,c]))
    if len(points) == 0:
        return np.zeros((0,3))
    points = np.vstack(points)
    return points

def point_cloud_to_volume_v2_batch(point_clouds, vsize=12, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxVxVxVxnum_samplex3
        Added on Feb 19
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume_v2(point_clouds[b,:,:], vsize, radius, num_sample)
        vol_list.append(np.expand_dims(vol, 0))
    return np.concatenate(vol_list, 0)

def point_cloud_to_volume_v2(points, vsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is vsize*vsize*vsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each voxel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    vol = np.zeros((vsize,vsize,vsize,num_sample,3))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n,:])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n,:])
    #print loc2pc

    for i in range(vsize):
        for j in range(vsize):
            for k in range(vsize):
                if (i,j,k) not in loc2pc:
                    vol[i,j,k,:,:] = np.zeros((num_sample,3))
                else:
                    pc = loc2pc[(i,j,k)] # a list of (3,) arrays
                    pc = np.vstack(pc) # kx3
                    # Sample/pad to num_sample points
                    if pc.shape[0]>num_sample:
                        choices = np.random.choice(pc.shape[0], num_sample, replace=False)
                        pc = pc[choices,:]
                    elif pc.shape[0]<num_sample:
                        pc = np.lib.pad(pc, ((0,num_sample-pc.shape[0]),(0,0)), 'edge')
                    # Normalize
                    pc_center = (np.array([i,j,k])+0.5)*voxel - radius
                    #print 'pc center: ', pc_center
                    pc = (pc - pc_center) / voxel # shift and scale
                    vol[i,j,k,:,:] = pc 
                #print (i,j,k), vol[i,j,k,:,:]
    return vol

def point_cloud_to_image_batch(point_clouds, imgsize, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxIxIxnum_samplex3
        Added on Feb 19
    """
    img_list = []
    for b in range(point_clouds.shape[0]):
        img = point_cloud_to_image(point_clouds[b,:,:], imgsize, radius, num_sample)
        img_list.append(np.expand_dims(img, 0))
    return np.concatenate(img_list, 0)


def point_cloud_to_image(points, imgsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is imgsize*imgsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each pixel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    img = np.zeros((imgsize, imgsize, num_sample, 3))
    pixel = 2*radius/float(imgsize)
    locations = (points[:,0:2] + radius)/pixel # Nx2
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n,:])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n,:])
    for i in range(imgsize):
        for j in range(imgsize):
            if (i,j) not in loc2pc:
                img[i,j,:,:] = np.zeros((num_sample,3))
            else:
                pc = loc2pc[(i,j)]
                pc = np.vstack(pc)
                if pc.shape[0]>num_sample:
                    choices = np.random.choice(pc.shape[0], num_sample, replace=False)
                    pc = pc[choices,:]
                elif pc.shape[0]<num_sample:
                    pc = np.lib.pad(pc, ((0,num_sample-pc.shape[0]),(0,0)), 'edge')
                pc_center = (np.array([i,j])+0.5)*pixel - radius
                pc[:,0:2] = (pc[:,0:2] - pc_center)/pixel
                img[i,j,:,:] = pc
    return img

def surface_normal_area(face, vertex):
    normals = list()
    areas = list()
    vertex_to_face = [[] for i in range(len(vertex))]
    for fid, f in enumerate(face):
        f = f[0]
        va, vb, vc = f[0], f[1], f[2]
        vertex_to_face[va].append(fid)
        vertex_to_face[vb].append(fid)
        vertex_to_face[vc].append(fid)

        a = vertex[vb] - vertex[va]
        b = vertex[vc] - vertex[va]
        normal = np.cross(a, b)
        area = np.dot(normal, normal) / 2.0
        normalized_normal = normal / np.linalg.norm(normal)
        normals.append(normalized_normal)
        areas.append(area)
    return np.array(normals), np.array(areas), vertex_to_face


def vertex_normal(vertex_to_face, normal, areas):
    vertex_normals = list()
    num_vertex = len(vertex_to_face)
    for vid in range(num_vertex):
        adj_faces = vertex_to_face[vid]
        if len(adj_faces)==0:  # single point with no adjancy points
            vertex_normals.append([0,0,1])
            continue
        adj_faces_area = np.expand_dims(np.array(areas[adj_faces]), axis=-1)
        adj_faces_normal = np.array(normal[adj_faces])
        avg_normal = (adj_faces_normal * adj_faces_area) / np.sum(adj_faces_area)
        avg_normal = np.sum(avg_normal, axis=0)
        normalized_normal = avg_normal / np.linalg.norm(avg_normal)
        #if np.isclose(np.linalg.norm(avg_normal), 0.0):
        #    print('-------------------')
        #    print(len(adj_faces))
        #    print('-------------------')
        #    print('-------------------')
        #    print(adj_faces_area.shape, adj_faces_normal.shape, adj_faces_area, adj_faces_normal) 
        #    print(adj_faces_normal * adj_faces_area)
        #    print(np.sum(adj_faces_area))
        #    print((adj_faces_normal * adj_faces_area) / np.sum(adj_faces_area))
        #    print(avg_normal, np.linalg.norm(avg_normal), adj_faces_area, adj_faces_normal) 
        #    print('-------------------')
        vertex_normals.append(normalized_normal)
    return np.array(vertex_normals)

# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def read_ply_rgba(filename):
    """ read XYZRGBA point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z,r,g,b,a] for x,y,z,r,g,b,a in pc])
    return pc_array

def read_ply_rgba_normal(filename):
    """ read XYZRGBA and NxNyNz point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z,r,g,b,a] for x,y,z,r,g,b,a in pc])
    face = plydata['face'].data
    f_n, f_a, v_f = surface_normal_area(face, pc_array[:, 0:3])
    v_n = vertex_normal(v_f, f_n, f_a)
    pc_array = np.concatenate((pc_array, v_n), axis=-1)
    return pc_array

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_ply_rgb(points, colors, filename, text=True):
    """ input: Nx3, Nx3 write points and colors to filename as PLY format. """
    num_points = len(points)
    assert len(colors) == num_points

    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    colors = [(colors[i,0], colors[i,1], colors[i,2]) for i in range(colors.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    color = np.array(colors, dtype=[('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    
    
    vertex_all = np.empty(num_points, vertex.dtype.descr + color.dtype.descr)
    
    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]
    
    for prop in color.dtype.names:
        vertex_all[prop] = color[prop]

    el = PlyElement.describe(vertex_all, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

def write_ply_rgb_normal(points, colors, normals, filename, text=True):
    """ input: Nx3, Nx3, Nx3 write points and colors to filename as PLY format. """
    num_points = len(points)
    assert len(colors) == num_points

    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    colors = [(colors[i,0], colors[i,1], colors[i,2]) for i in range(colors.shape[0])]
    normals = [(normals[i,0], normals[i,1], normals[i,2]) for i in range(normals.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    color = np.array(colors, dtype=[('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    normal = np.array(normals, dtype=[('nx', 'f4'), ('ny', 'f4'),('nz', 'f4')])
    
    
    vertex_all = np.empty(num_points, vertex.dtype.descr + color.dtype.descr + normal.dtype.descr)
    
    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]
    
    for prop in color.dtype.names:
        vertex_all[prop] = color[prop]

    for prop in normal.dtype.names:
        vertex_all[prop] = normal[prop]

    el = PlyElement.describe(vertex_all, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)
# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0,1,2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter-1)/2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i-radius) + (j-radius) * (j-radius) <= radius * radius:
                disk[i, j] = np.exp((-(i-radius)**2 - (j-radius)**2)/(radius**2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]
    
    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])
       
    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize/2 + (x*space)
        yc = canvasSize/2 + (y*space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))
        
        px = dx + xc
        py = dy + yc
        
        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
    
    image = image / np.max(image)
    return image

def point_cloud_three_views(points):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """ 
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    img1 = draw_point_cloud(points, zrot=110/180.0*np.pi, xrot=45/180.0*np.pi, yrot=0/180.0*np.pi)
    img2 = draw_point_cloud(points, zrot=70/180.0*np.pi, xrot=135/180.0*np.pi, yrot=0/180.0*np.pi)
    img3 = draw_point_cloud(points, zrot=180.0/180.0*np.pi, xrot=90/180.0*np.pi, yrot=0/180.0*np.pi)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large


def point_cloud_three_views_demo():
    """ Demo for draw_point_cloud function """
    from PIL import Image
    points = read_ply('../third_party/mesh_sampling/piano.ply')
    im_array = point_cloud_three_views(points)
    img = Image.fromarray(np.uint8(im_array*255.0))
    img.save('piano.jpg')

if __name__=="__main__":
    point_cloud_three_views_demo()

def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #savefig(output_filename)

def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)

def write_ply_color(points, labels, out_filename, num_classes=None, colors=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    import matplotlib.pyplot as pyplot
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
        print(num_classes)
    else:
        assert(num_classes>np.max(labels))
    if colors is None:
        #colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
        colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[labels[i]]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()


def farthest_pts_sampling_abuse(pts, num_samples):
    '''
    naive method
    :param pts: n x 3 ndarray
    :param num_samples:
    :return: num_samples x 3 ndarray
    '''
    diff = pts[:, None, :] - pts[None, :, :]
    # dis_mat = np.sum(diff * diff, axis=2)
    dis_mat = np.linalg.norm(diff, axis=2)
    N = num_samples

    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = dis_mat[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, dis_mat[idx, :])
    return pts[perm, :]


def farthest_pts_sampling(coords, num_samples):
    '''
    naive method
    :param pts: n x 3 ndarray
    :param num_samples:
    :return: num_samples x 3 ndarray
    '''
    pts = coords.numpy()
    dis_mat = np.linalg.norm(pts, axis=2)

    point_set = []

    perm = np.zeros(num_samples, dtype=np.int64)
    index = random.randint(0, pts.shape[0] - 1)
    point_set.append(pts[index])
    pts[index] = np.array([-10000, -10000, -10000])
    for i in range(1, num_samples):
        refer = pts[index]
        diff = np.linalg.norm(pts[:, :] - refer[None, :], axis=1)
        index = np.argmin(diff)
        point_set.append(pts[index])
        pts[index] = np.array([-10000, -10000, -10000])

    point_set = np.vstack(point_set)
    return point_set

def random_partition(coords):
    # print('1')
    mask = torch.ones(coords.size()[0]).numpy()
    coords_np = coords.numpy()
    sample_num = random.randint(2, 5)

    random_index = np.random.randint(coords_np.shape[0], size=sample_num)
    sample_points = coords_np[random_index, :]

    diff = coords_np[:, None, :] - sample_points[None, :, :]
    diff = np.linalg.norm(diff, axis=2)
    partitions = np.argmin(diff, axis=1)
    filter_ind = random.randint(0, sample_num - 1)

    # coords_torch = torch.from_numpy(coords_np[partitions != filter_ind])
    coords_torch = coords

    mask[partitions == filter_ind] = 0
    mask = torch.from_numpy(mask)
    # print('4')
    # part1 = torch.from_numpy(coords_np[partitions == filter_ind])
    # part2 = torch.from_numpy(coords_np[partitions != filter_ind])
    return coords_torch, mask
    # return part1, part2

def random_rotation(coords):
    # scale = torch.eye(3)*random.uniform(0.95, 1.05)
    scale_flip = np.eye(3) + np.random.randn(3, 3) * 0.1
    scale_flip[0][0] *= np.random.randint(0, 2) * 2 - 1
    scale_flip = torch.from_numpy(scale_flip).float()

    # scale = torch.eye(3)
    theta = random.uniform(0, 2) * math.pi
    rotationx = torch.tensor([[math.cos(theta), math.sin(theta), 0],
                             [-math.sin(theta), math.cos(theta), 0],
                             [0, 0, 1]]).float()

    # rotationy = torch.tensor([[math.cos(theta), 0, math.sin(theta)],
    #                          [0, 1, 0],
    #                          [math.sin(theta), 0, -math.cos(theta)]]).float()
    #
    # rotationz = torch.tensor([[1, 0, 0],
    #                           [0, math.cos(theta), math.sin(theta)],
    #                           [0, -math.sin(theta), math.cos(theta)]]).float()

    m = torch.matmul(scale_flip, rotationx)
    coords = torch.matmul(coords.float(), m)
    return coords


# def random_rotation(coords):
#     return coords

def resize_rotation(coords, item):
    scale = 0

    if item == 'chair':
        scale = torch.eye(3) * 0.8
    elif item == 'sofa':
        scale = torch.eye(3) * 1.75
    elif item == 'table':
        scale = torch.eye(3) * 1.65
    elif item == 'bookshelf':
        scale = torch.eye(3) * 1.7
    elif item == 'desk':
        scale = torch.eye(3) * 1.25
    elif item == 'bed':
        scale = torch.eye(3) * 2.1
    elif item == 'sink':
        scale = torch.eye(3) * 1.05
    elif item == 'bathtub':
        scale = torch.eye(3) * 1.25
    elif item == 'toilet':
        scale = torch.eye(3) * 0.65
    elif item == 'door':
        scale = torch.eye(3) * 1.8
    elif item == 'curtain':
        scale = torch.eye(3) * 2
    else :
        scale = torch.eye(3) * random.uniform(0.9, 1.75)

    '''
    if item == 'chair':
        scale = torch.eye(3) * random.uniform(5, 5.5)
    elif item == 'bed':
        scale = torch.eye(3) * random.uniform(1.4, 1.6)
    elif item == 'sofa':
        scale = torch.eye(3) * random.uniform(9, 9.5)
    elif item == 'table':
        scale = torch.eye(3) * random.uniform(8, 8.5)
    elif item == 'bookshelf':
        scale = torch.eye(3) * random.uniform(1.1, 1.2)
    elif item == 'desk':
        scale = torch.eye(3) * random.uniform(7, 7.5)
    elif item == 'nega_data':
        scale = torch.eye(3) * random.uniform(5, 8)
    '''
    # theta = 0 * math.pi

    # rotationx = torch.tensor([[math.cos(theta), math.sin(theta), 0],
    #                          [-math.sin(theta), math.cos(theta), 0],
    #                          [0, 0, 1]]).float()
    #
    # rotationy = torch.tensor([[math.cos(theta), 0, math.sin(theta)],
    #                          [0, 1, 0],
    #                          [math.sin(theta), 0, -math.cos(theta)]]).float()

    # rotationz = torch.tensor([[1, 0, 0],
    #                           [0, math.cos(theta), math.sin(theta)],
    #                           [0, -math.sin(theta), math.cos(theta)]]).float()

    # m = torch.matmul(scale, rotationz)
    m = scale
    coords = torch.matmul(coords.float(), m)

    return coords
import numpy as np 
from numpy.linalg import lstsq, inv
import math 
from PIL import Image

INTRINSIC_DIR = '2.1-2.5/intrinsics/'
EXTRINSIC_DIR = '2.1-2.5/extrinsics/'
RGB_DIR = '2.1-2.5/rgb/'

def load_params(intrinsic_file, extrinsic_file):
	intrinsic = open(intrinsic_file, 'r')
	extrinsic = open(extrinsic_file, 'r')

	intrinsic_matrix = list()

	for i in intrinsic:
		arr = [float(j) for j in i.split(' ')]
		intrinsic_matrix.append(arr)

	for e in extrinsic:
		extrinsic_vector = [float(k) for k in e.split(',')]

	return np.array(intrinsic_matrix), np.array(extrinsic_vector)

def generate_point_cloud(rgb_file, intrinsic, extrinsic): 
	'''
	generate point cloud given rgbd image and camera intrinsic and extrinsic

	input: rgb (colored image)
			intrinsic (3 x 3 matrix): includes camera focals etc 
			extrinsic (6 x 1 vector): x,y,z,rx,ry,rz
	output: n x 6 array (x,y,z,r,g,b)
	'''
	def get_rt(extrinsic):
		x, y, z, rx, ry, rz = extrinsic 

		rotation = np.array([[math.cos(rx)*math.cos(ry), 
							math.cos(rx)*math.sin(ry)*math.sin(rz) - math.sin(rx)*math.cos(rz),
							math.cos(rx)*math.sin(ry)*math.cos(rz) + math.sin(rx)*math.sin(rz)],
							[math.sin(rx)*math.cos(ry), 
							math.sin(rx)*math.sin(ry)*math.sin(rz) + math.cos(rx)*math.cos(rz),
							math.sin(rx)*math.sin(ry)*math.cos(rz) - math.cos(rx)*math.sin(rz)],
							[-math.sin(ry), 
							math.cos(ry)*math.sin(rz), 
							math.cos(ry)*math.cos(rz)]])

		translation = np.array([x, y, z])

		return rotation, translation

	rgb = Image.open(rgb_file)

	point_cloud = list() 

	rotation, translation = get_rt(extrinsic)
	k_inverse = inv(intrinsic)
	r_transpose = np.transpose(rotation)

	for v in range(rgb.size[1]):
		for u in range(rgb.size[0]):

			img = np.array([u,v,1])

			x, y, z = np.matmul(r_transpose, np.dot(k_inverse, img) - translation)
			r, g, b = rgb.getpixel((u,v))

			point_cloud.append([x, y, z, r, g, b])

	return np.array(point_cloud)


def point_cloud_surface_normal(point_cloud, pos):
	pass 

sample0 = '0.txt'
sample1 = '1.txt'

rgb0 = '0.png'
rgb1 = '1.png'

intrinsic, extrinsic = load_params(INTRINSIC_DIR+sample0, EXTRINSIC_DIR+sample0)
generate_point_cloud(RGB_DIR+rgb0, intrinsic, extrinsic)

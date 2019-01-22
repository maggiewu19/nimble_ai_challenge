'''
Kinematics Challenge
'''

import numpy as np
import math3d as m3d
import math 
import copy
from scipy import optimize


class robot(object):
	'''
	robot is a class for kinematics and control for a robot arm
	'''

	def __init__(self, base_frame=m3d.Transform, tool_transform=m3d.Transform):
		self.base_frame = base_frame
		self.tool_transform = tool_transform

		# arm length (d-h)
		a_2 = -0.425
		a_3 = -0.39225 
		d_1 = 0.089159
		d_4 = 0.10915
		d_5 = 0.09465 
		d_6 = 0.0823 
		self.alpha_minus_1 = np.array([0, np.pi/2, 0, 0, np.pi/2, -np.pi/2])
		self.a_minus_1 = np.array([0, 0, a_2, a_3, 0, 0])
		self.d = np.array([d_1, 0, 0, d_4, d_5, d_6])

	def transformation(t, alpha, a, d): 
		''' To 
			
			input: t (theta) in radians 
				   alpha from self.alpha_minus_1
				   a from self.a_minus_1
				   d from self.d 
		'''
		return m3d.Transform(np.array([[math.cos(t), -math.sin(t), 0, a], 
										[math.sin(t)*math.cos(alpha), math.cos(t)*math.cos(alpha), -math.sin(alpha), -math.sin(alpha)*d], 
										[math.sin(t)*math.sin(alpha), math.cos(t)*math.sin(alpha), math.cos(alpha), math.cos(alpha)*d], 
										[0, 0, 0, 1]]))

	# FORWARD KINEMATICS:
	def getFK(self, joint_angles):
		''' To calculate the forward kinematics input joint_angles which is a list or numpy array of 6 joint angles

			input = joint angles in degrees
		'''

		tool_frame = m3d.Transform()
		self.thetas = joint_angles 

		t, alpha, a, d = self.thetas[0], self.alpha_minus_1[0], self.a_minus_1[0], self.d[0]
		T_0_6 = self.transformation(t, alpha, a, d)

		for i in range(1, 6): 
			t, alpha, a, d = self.thetas[i], self.alpha_minus_1[i], self.a_minus_1[i], self.d[i]
			T_0_6 *= self.transformation(t, alpha, a, d)

		tool_frame.set_orient(T_0_6.get_orient())
		tool_frame.set_pos(T_0_6.get_pos())

		print ('Forward Kinematics', tool_frame)

		return tool_frame


	def getJacobian(self, joint_angles, num_frames=1, epsilon=1e-6):
		'''
		Numerically calculates the 6 x 6 Jacobian matrix which relates joint torques/velocities
		to end-effector forces-torque wrench and velocity.  The first 3 rows of the
		Jacobian are the translational partial derivatives while the last 3 rows are
		the rotational partial derivatives with respect to each joint.

		input: joint_angles = angles in degrees
		'''

		# Convert epsilon into radians
		epsilon_rad = epsilon*np.pi/180

		# find number of joints in the robot
		num_joints = len(joint_angles)
		jacobian = np.zeros((6,num_joints))

		# solve for the forward kinematics of the original joint angles
		tool = self.getFK(joint_angles)

		''' loop through each joint and perturb just that joint angle by epsilon and recalculate
			the FK and take the numerical derivative  '''
		for i in range(num_joints):

			joint_angles_perturb = copy.deepcopy(joint_angles)

			# perturb just joint i
			joint_angles_perturb[i] += epsilon

			# recalculate FK with the ith joint angle perturbed
			tool_perturb = self.getFK(joint_angles_perturb)

			# compute numerical derivative to populate jacobian matrix
			for j in range(num_frames):

				# calculate translational partial derivatives in world frame
				jacobian[6*j:6*j+3, i] = (tool_perturb[:3,3] - tool[:3,3])/epsilon_rad

				# calculate rotational partial derivative in joint frame
				W = np.matmul(tool[:3,:3].T, tool_perturb[:3,:3])

				# convert rotational partials to world frame
				dFK_rot = np.zeros((3,))
				dFK_rot[0] = (-W[1,2] + W[2,1])/(2*epsilon_rad)
				dFK_rot[1] = (-W[2,0] + W[0,2])/(2*epsilon_rad)
				dFK_rot[2] = (-W[0,1] + W[1,0])/(2*epsilon_rad)

				# populate jacobian with the rotational derivative
				jacobian[6*j+3:6*j+6, i] = np.matmul(tool[:3,:3], dFK_rot.T)

		# assign the class instance jacobian value
		self.jacobian = jacobian

		return jacobian


	# def getIK(self, poseGoal, seed_joint_angles=np.zeros((6,))):
	# 	''' Analytically solve the inverse kinematics

	# 		inputs: poseGoal = [x,y,z,rx,ry,rz] goal position of end-effector in global frame.
	# 							Orientation [rx,ry,rz] specified in rotation vector (axis angle) form [rad]
	# 				seed_joint_angles = [theta1, theta2, theta3, theta4, theta5, theta6]
	# 									 seed angles for comparing solutions to ensure mode change does not occur

	# 		outputs: joint_angles = joint angles in rad to reach goal end-effector pose
	# 	'''

	# 	joint_angles = np.zeros((1,6))
	# 	####  YOUR CODE GOES HERE

	# 	return joint_angles

	def getIK(self, poseTransform, seed_joint_angles=np.zeros((6,))):
		''' Analytically solve the inverse kinematics

			inputs: poseTransform = 4 x 4 end effector transformation matrix
					seed_joint_angles = [theta1, theta2, theta3, theta4, theta5, theta6]
										 seed angles for comparing solutions to ensure mode change does not occur

			outputs: joint_angles = joint angles in rad to reach goal end-effector pose
		'''

		joint_angles = np.zeros((1,6))

		print ('Pose Transform', poseTransform)
		P_0_6 = poseTransform.get_pos()
		P_0_6x = P_0_6[0]
		P_0_6y = P_0_6[1]

		O_0_6 = poseTransform.get_orient()
		X_0_6x = O_0_6[0][0]
		X_0_6y = O_0_6[1][0]
		Y_0_6x = O_0_6[0][1]
		Y_0_6y = O_0_6[1][1]

		P_0_5 = poseTransform * m3d.Vector(np.array([0, 0, -self.d[5]]))
		P_0_5x = P_0_5[0]
		P_0_5y = P_0_5[1]

		# theta 1 (2 possible solutions)
		theta1 = [math.atan2(P_0_5y, P_0_5x) + math.acos(self.d[3]/math.sqrt(P_0_5x**2 + P_0_5y**2)) + np.pi/2, 
				math.atan2(P_0_5y, P_0_5x) - math.acos(self.d[3]/math.sqrt(P_0_5x**2 + P_0_5y**2)) + np.pi/2]
		print ('Theta 1', theta1)

		# theta 5 (2 possible solutions)
		theta5 = [(math.acos(P_0_6x * math.sin(theta1) - P_0_6y * math.cos(theta1) - self.d[3]))/self.d[5], 
				-(math.acos(P_0_6x * math.sin(theta1) - P_0_6y * math.cos(theta1) - self.d[3]))/self.d[5]]
		print ('Theta 5', theta5)

		# theta 6 (1 possible solution)
		theta6 = [math.atan2((-X_0_6y * math.sin(theta1) + Y_0_6y * math.cos(theta1))/math.sin(theta5), (X_0_6x * math.sin(theta1) - Y_0_6x * math.cos(theta1))/math.sin(theta5))]
		print ('Theta 6', theta6)

		# theta 3 (2 possible solutions)
		rotation_0_1 = m3d.Orientation(np.array([[math.cos(theta1), math.sin(theta1), 0],
												[-math.sin(theta1), math.cos(theta1), 0], 
												[0, 0, 1]])) 
		P_1_5 = rotation_0_1 * P_0_5 
		T_5_6 = self.transformation(theta6, self.alpha_minus_1[5], self.a_minus_1[5], self.d[5])
		T_4_5 = self.transformation(theta5, self.alpha_minus_1[4], self.a_minus_1[4], self.d[4])
		T_0_1 = self.transformation(theta1, self.alpha_minus_1[0], self.a_minus_1[0], self.d[0])

		T_1_4 = T_0_1.get_inverse() * poseTransform * T_5_6.get_inverse() * T_4_5.get_inverse() 

		P_1_4 = T_1_4.get_pos() 
		P_1_4x = P_1_4[0]
		P_1_4y = P_1_4[1]
		P_1_4z = P_1_4[2]

		theta3 = [math.acos((P_1_4x**2+P_1_4z**2-self.a_minus_1[2]**2-self.a_minus_1[3]**2)/(2*self.a_minus_1[2]*self.a_minus_1[3])), 
				-math.acos((P_1_4x**2+P_1_4z**2-self.a_minus_1[2]**2-self.a_minus_1[3]**2)/(2*self.a_minus_1[2]*self.a_minus_1[3]))]
		print ('Theta 3', theta3)

		# theta 2 (1 possible solution)
		theta2 = [math.atan2(-P_1_4z, -P_1_4x)-math.asin(-self.a_minus_1[3]*math.sin(theta3)/math.sqrt(P_1_4x**2+P_1_4z**2))]
		print ('Theta 2', theta2)

		# theta 4 (1 possible solution)
		T_1_2 = self.transformation(theta2, self.alpha_minus_1[1], self.a_minus_1[1], self.d[1])
		T_2_3 = self.transformation(theta3, self.alpha_minus_1[2], self.a_minus_1[2], self.d[2])

		T_3_4 = T_2_3.get_inverse() * T_1_2.get_inverse() * T_1_4 

		O_3_4 = T_3_4.get_orient()
		X_3_4x = O_3_4[0][0]
		X_3_4y = O_3_4[1][0]

		theta = [math.atan2(X_3_4y, X_3_4x)]
		print ('Theta 4', theta4)

		# select best solution out of valid thetas 

		return joint_angles 


	def xyzToolAngleError(self, joint_angles, poseGoal, angleErrorScale=.02):
		''' Calculate the error between the goal position and orientation and the actual
			position and orientation

			inputs: poseGoal = [x,y,z,rx,ry,rz] position of goal. orientation of goal specified in rotation vector (axis angle) form [rad]
					joint_angles = current joint angles from optimizer [rad]
					angleErrorScale = coefficient to determine how much to weight orientation in
									  the optimization. value of ~ .05 has equal weight to position
		'''

		####  YOUR CODE GOES HERE
		totalError = 0

		return totalError


	def getIKnum(self, xyzGoal, eulerAnglesGoal, seed_joint_angles=np.zeros((6,))):
		''' Numerically calculate the inverse kinematics through constrained optimization

			inputs: poseGoal = [x,y,z,rx,ry,rz] goal position of end-effector in global frame.
								Orientation [rx,ry,rz] specified in rotation vector (axis angle) form [rad]
					seed_joint_angles = [theta1, theta2, theta3, theta4, theta5, theta6]
										 seed angles for comparing solutions to ensure mode change does not occur

			outputs: joint_angles = joint angles in rad to reach goal end-effector pose
		'''
		joint_angles = np.zeros((1,6))
		####  YOUR CODE GOES HERE

		return joint_angles


tool_transform = m3d.Transform(np.array([[1, 0, 0, 0], 
										[0, 1, 0, 0],
										[0, 0, 1, 1],
										[0, 0, 0, 1]])) 
ur5 = robot(tool_transform=tool_transform) 
ur5fk = ur5.getFK(np.array([np.pi/2, np.pi/2, 0, 0, np.pi/2, 0]))
ur5.getIK(ur5fk)



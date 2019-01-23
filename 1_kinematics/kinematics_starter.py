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

	def transformation(self, t, alpha, a, d): 
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

	def getIK(self, poseTransform, seed_joint_angles=np.zeros((6,)), zero_limit=1e-5, infty_limit=1e10, err=float('inf')):
		''' Analytically solve the inverse kinematics

			inputs: poseTransform = 4 x 4 end effector transformation matrix
					seed_joint_angles = [theta1, theta2, theta3, theta4, theta5, theta6]
										 seed angles for comparing solutions to ensure mode change does not occur

			outputs: joint_angles = joint angles in rad to reach goal end-effector pose
		'''

		joint_angles = np.zeros((1,6))

		print ('Pose Transform', poseTransform)

		P_0_6 = poseTransform.get_pos()
		O_0_6 = poseTransform.get_orient()
		P_0_5 = poseTransform * m3d.Vector(np.array([0, 0, -self.d[5]]))

		def angle_restriction(angle):
			if -zero_limit < angle < zero_limit or 2*np.pi - zero_limit < angle < 2*np.pi + zero_limit:
				return 0 

			while angle > 2*np.pi: 
				angle -= 2*np.pi

			while angle < 0: 
				angle += 2*np.pi 

			return angle 

		# theta 1 (2 possible solutions)
		def get_theta1(P_0_5): 
			P_0_5x = P_0_5[0]
			P_0_5y = P_0_5[1]

			acos_val = self.d[3]/math.sqrt(P_0_5x**2 + P_0_5y**2)

			if 1 + zero_limit > abs(acos_val) > 1 - zero_limit: 
				acos_val = 1 

			if abs(acos_val) > 1: 
				return None 

			sol1 = angle_restriction(math.atan2(P_0_5y, P_0_5x) + math.acos(acos_val) + np.pi/2)
			sol2 = angle_restriction(math.atan2(P_0_5y, P_0_5x) - math.acos(acos_val) + np.pi/2)

			theta1 = [sol1, sol2]

			# print ('Theta 1', theta1)

			return theta1 

		# theta 5 (2 possible solutions)
		def get_theta5(s, P_0_6):
			theta1 = s[0]

			P_0_6 = P_0_6[0]

			P_0_6x = P_0_6[0]
			P_0_6y = P_0_6[1]

			acos_val = (P_0_6x * math.sin(theta1) - P_0_6y * math.cos(theta1) - self.d[3])/self.d[5]

			if 1 + zero_limit > abs(acos_val) > 1 - zero_limit: 
				acos_val = 1 

			if abs(acos_val) > 1 + zero_limit: 
				return None 

			sol1 = angle_restriction(math.acos(acos_val))
			sol2 = angle_restriction(-math.acos(acos_val))

			theta5 = [sol1, sol2]

			# print ('Theta 5', theta5)

			return theta5 

		# theta 6 (1 possible solution)
		def get_theta6(s, O_0_6):
			theta1, theta5 = s

			O_0_6 = O_0_6[0]

			X_0_6x = O_0_6[0][0]
			X_0_6y = O_0_6[1][0]
			Y_0_6x = O_0_6[0][1]
			Y_0_6y = O_0_6[1][1]

			sol1 = angle_restriction(math.atan2((-X_0_6y * math.sin(theta1) + Y_0_6y * math.cos(theta1))/math.sin(theta5), (X_0_6x * math.sin(theta1) - Y_0_6x * math.cos(theta1))/math.sin(theta5)))

			theta6 = [sol1]
			# print ('Theta 6', theta6)

			return theta6 

		def get_T_1_4(theta1, theta5, theta6, P_0_5):
			rotation_0_1 = m3d.Orientation(np.array([[math.cos(theta1), math.sin(theta1), 0],
													[-math.sin(theta1), math.cos(theta1), 0], 
													[0, 0, 1]])) 
			P_1_5 = rotation_0_1 * P_0_5 
			T_5_6 = self.transformation(theta6, self.alpha_minus_1[5], self.a_minus_1[5], self.d[5])
			T_4_5 = self.transformation(theta5, self.alpha_minus_1[4], self.a_minus_1[4], self.d[4])
			T_0_1 = self.transformation(theta1, self.alpha_minus_1[0], self.a_minus_1[0], self.d[0])

			T_1_4 = T_0_1.get_inverse() * poseTransform * T_5_6.get_inverse() * T_4_5.get_inverse() 
			# print ('T_1_4', T_1_4)

			return T_1_4

		# theta 3 (2 possible solutions)
		def get_theta3(s, P_0_5): 
			theta1, theta5, theta6 = s

			P_0_5 = P_0_5[0]

			T_1_4 = get_T_1_4(theta1, theta5, theta6, P_0_5)

			P_1_4 = T_1_4.get_pos() 
			P_1_4x = P_1_4[0]
			P_1_4z = P_1_4[2] 

			acos_val = (P_1_4x**2+P_1_4z**2-self.a_minus_1[2]**2-self.a_minus_1[3]**2)/(2*self.a_minus_1[2]*self.a_minus_1[3])

			if 1 + zero_limit > abs(acos_val) > 1 - zero_limit: 
				acos_val = 1 

			if abs(acos_val) > 1: 
				return None 

			sol1 = angle_restriction(math.acos(acos_val))
			sol2 = angle_restriction(-math.acos(acos_val))

			theta3 = [sol1, sol2]
			# print ('Theta 3', theta3)

			return theta3 

		# theta 2 (1 possible solution)
		def get_theta2(s, P_0_5): 
			theta1, theta5, theta6, theta3 = s

			P_0_5 = P_0_5[0]

			T_1_4 = get_T_1_4(theta1, theta5, theta6, P_0_5)

			P_1_4 = T_1_4.get_pos() 
			P_1_4x = P_1_4[0]
			P_1_4z = P_1_4[2] 

			sol1 = angle_restriction(math.atan2(-P_1_4z, -P_1_4x)-math.asin(-self.a_minus_1[3]*math.sin(theta3)/math.sqrt(P_1_4x**2+P_1_4z**2)))

			theta2 = [sol1]
			# print ('Theta 2', theta2)

			return theta2 

		def get_T_3_4(theta2, theta3, T_1_4):
			T_1_2 = self.transformation(theta2, self.alpha_minus_1[1], self.a_minus_1[1], self.d[1])
			T_2_3 = self.transformation(theta3, self.alpha_minus_1[2], self.a_minus_1[2], self.d[2])

			T_3_4 = T_2_3.get_inverse() * T_1_2.get_inverse() * T_1_4 
			return T_3_4

		# theta 4 (1 possible solution)
		def get_theta4(s, P_0_5):
			theta1, theta5, theta6, theta3, theta2 = s

			P_0_5 = P_0_5[0]

			T_1_4 = get_T_1_4(theta1, theta5, theta6, P_0_5)
			T_3_4 = get_T_3_4(theta2, theta3, T_1_4)

			O_3_4 = T_3_4.get_orient()
			X_3_4x = O_3_4[0][0]
			X_3_4y = O_3_4[1][0]

			sol1 = angle_restriction(math.atan2(X_3_4y, X_3_4x))

			theta4 = [sol1]
			# print ('Theta 4', theta4)

			return theta4 

		def get_joint_angle(solutions, f, *args):
			next_s = list() 

			for s in solutions: 
				theta = f(s, args)

				if theta != None: 
					# more than one solution 
					for t in theta: 
						next_s.append(s + [t])

			return next_s 

		def rmse(solution, seed):
			diff = solution - seed
			mse = sum(np.power(diff, 2))
			return math.sqrt(mse)

		# get all valid solutions 
		# (theta1, theta5, theta6, theta3, theta2, theta4)
		theta1 = get_theta1(P_0_5) 

		if theta1 == None: 
			print ('IK cannot be found!')
			return None 
		
		theta1 = [[theta1[0]], [theta1[1]]]
		theta5 = get_joint_angle(theta1, get_theta5, P_0_6)
		theta6 = get_joint_angle(theta5, get_theta6, O_0_6)
		theta3 = get_joint_angle(theta6, get_theta3, P_0_5)
		theta2 = get_joint_angle(theta3, get_theta2, P_0_5)
		theta4 = get_joint_angle(theta2, get_theta4, P_0_5)

		# correct ordering
		solutions = list() 
		for s in theta4: 
			new_s = np.array([s[0], s[4], s[3], s[5], s[1], s[2]])
			solutions.append(new_s)

		# select best solution out of valid thetas 
		for s in solutions: 
			error = rmse(s, seed_joint_angles)
			if error < err: 
				error = err 
				joint_angles = s 

		print (joint_angles)

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
ur5.getIK(ur5fk, seed_joint_angles=np.array([1.5707963267948957, 1.5707963267948966, 0, 0, 1.5707963267948966, 0]))



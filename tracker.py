import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import math

class KalmanFilter:
    def __init__(self, dt=1, u_x=1, u_y=1, std_acc=0.2, x_std_meas=0.1, y_std_meas=0.1):
        # Time step
        self.dt = dt

        # Initial State [x, y, v_x, v_y]
        self.x = np.matrix([[0], [0], [0], [0]])

        # Control input (acceleration)
        self.u = np.matrix([[u_x], [u_y]])

        # State transition matrix (applies the time step to the velocities)
        self.F = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Control input matrix (maps acceleration to position changes)
        self.B = np.matrix([[self.dt**2 / 2, 0],
                            [0, self.dt**2 / 2],
                            [self.dt, 0],
                            [0, self.dt]])

        # Measurement matrix (maps the predicted position to the measurement space)
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Process noise covariance
        self.Q = np.matrix([[self.dt**4 / 4, 0, self.dt**3 / 2, 0],
                            [0, self.dt**4 / 4, 0, self.dt**3 / 2],
                            [self.dt**3 / 2, 0, self.dt**2, 0],
                            [0, self.dt**3 / 2, 0, self.dt**2]]) * std_acc**2

        # Measurement noise covariance
        self.R = np.matrix([[x_std_meas**2, 0],
                            [0, y_std_meas**2]])

        # Initial covariance matrix
        self.P = np.eye(self.F.shape[1])

    def predict(self):
        # Predict the state and the error covariance
        self.x = np.dot(self.F, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0:2]

    def update(self, z):
        # Compute the Kalman gain
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Update the estimate
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))

        # Update the error covariance
        I = np.eye(self.H.shape[1])
        self.P = (I - np.dot(K, self.H)) * self.P


class Tracker:
    def __init__(self, max_distance=50, max_disappeared=10, iou_threshold=0.3):
        self.objects = {}
        self.disappeared = {}
        self.id_count = 0
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.kalman_filters = {}

    def iou(self, boxA, boxB):
        # Calculate Intersection Over Union (IoU)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def update(self, objects_rect):
        if len(self.objects) == 0:
            # Register new objects
            for rect in objects_rect:
                self.register(rect)
        else:
            # Compute cost matrix (IoU)
            object_ids = list(self.objects.keys())
            existing_boxes = list(self.objects.values())

            cost_matrix = np.zeros((len(existing_boxes), len(objects_rect)))

            for i, existing_box in enumerate(existing_boxes):
                for j, new_box in enumerate(objects_rect):
                    cost_matrix[i, j] = 1 - self.iou(existing_box, new_box)

            # Solve the assignment problem
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            assigned_rows = set()
            assigned_cols = set()

            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 1 - self.iou_threshold:
                    object_id = object_ids[row]
                    self.objects[object_id] = objects_rect[col]
                    self.kalman_filters[object_id].update(np.array([[objects_rect[col][0]], [objects_rect[col][1]]]))
                    self.disappeared[object_id] = 0
                    assigned_rows.add(row)
                    assigned_cols.add(col)

            # Mark objects as disappeared if not assigned
            unassigned_rows = set(range(len(existing_boxes))) - assigned_rows
            for row in unassigned_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new objects for unassigned detections
            unassigned_cols = set(range(len(objects_rect))) - assigned_cols
            for col in unassigned_cols:
                self.register(objects_rect[col])

    def register(self, rect):
        # Register new object with a Kalman filter
        self.objects[self.id_count] = rect
        self.kalman_filters[self.id_count] = KalmanFilter()
        self.disappeared[self.id_count] = 0
        self.id_count += 1

    def deregister(self, object_id):
        # Deregister object and remove from the dictionary
        del self.objects[object_id]
        del self.kalman_filters[object_id]
        del self.disappeared[object_id]

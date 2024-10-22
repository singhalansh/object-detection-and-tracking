# import math


# class Tracker:
#     def __init__(self):
#         # Store the center positions of the objects
#         self.center_points = {}
#         # Keep the count of the IDs
#         # each time a new object id detected, the count will increase by one
#         self.id_count = 0


#     def update(self, objects_rect):
#         # Objects boxes and ids
#         objects_bbs_ids = []

#         # Get center point of new object
#         for rect in objects_rect:
#             x, y, w, h = rect
#             cx = (x + x + w) // 2
#             cy = (y + y + h) // 2

#             # Find out if that object was detected already
#             same_object_detected = False
#             for id, pt in self.center_points.items():
#                 dist = math.hypot(cx - pt[0], cy - pt[1])

#                 if dist < 35:
#                     self.center_points[id] = (cx, cy)
# #                    print(self.center_points)
#                     objects_bbs_ids.append([x, y, w, h, id])
#                     same_object_detected = True
#                     break

#             # New object is detected we assign the ID to that object
#             if same_object_detected is False:
#                 self.center_points[self.id_count] = (cx, cy)
#                 objects_bbs_ids.append([x, y, w, h, self.id_count])
#                 self.id_count += 1

#         # Clean the dictionary by center points to remove IDS not used anymore
#         new_center_points = {}
#         for obj_bb_id in objects_bbs_ids:
#             _, _, _, _, object_id = obj_bb_id
#             center = self.center_points[object_id]
#             new_center_points[object_id] = center

#         # Update dictionary with IDs not used removed
#         self.center_points = new_center_points.copy()
#         return objects_bbs_ids





import math

class Tracker:
    def __init__(self, max_distance=50, max_disappeared=5):
        # Store the center positions of the objects
        self.center_points = {}
        # To track how long an object has been "missing"
        self.disappeared = {}
        # Count of the IDs
        self.id_count = 0
        # Distance threshold to consider an object to be the same
        self.max_distance = max_distance
        # Number of frames an object can be missing before removal
        self.max_disappeared = max_disappeared

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new objects
        input_centers = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            input_centers.append((cx, cy, rect))

        # If we have no objects currently being tracked, initialize them
        if len(self.center_points) == 0:
            for center in input_centers:
                cx, cy, rect = center
                self.center_points[self.id_count] = (cx, cy)
                self.disappeared[self.id_count] = 0
                objects_bbs_ids.append([rect[0], rect[1], rect[2], rect[3], self.id_count])
                self.id_count += 1
        else:
            # Match new object centers to existing ones
            object_ids = list(self.center_points.keys())
            existing_centers = list(self.center_points.values())

            # Create a distance matrix between new and old objects
            distance_matrix = []
            for (new_cx, new_cy, _) in input_centers:
                row = []
                for (old_cx, old_cy) in existing_centers:
                    dist = math.hypot(new_cx - old_cx, new_cy - old_cy)
                    row.append(dist)
                distance_matrix.append(row)

            # Assign new objects to existing based on closest distance (greedy algorithm)
            used_rows = set()
            used_columns = set()
            for row in range(len(distance_matrix)):
                min_distance = min(distance_matrix[row])
                col = distance_matrix[row].index(min_distance)

                # If within the max distance, assign the object to that ID
                if min_distance < self.max_distance and col not in used_columns and row not in used_rows:
                    object_id = object_ids[col]
                    self.center_points[object_id] = (input_centers[row][0], input_centers[row][1])
                    self.disappeared[object_id] = 0
                    objects_bbs_ids.append([input_centers[row][2][0], input_centers[row][2][1], input_centers[row][2][2], input_centers[row][2][3], object_id])
                    used_columns.add(col)
                    used_rows.add(row)

            # Mark the remaining existing objects as disappeared
            unused_columns = set(range(len(existing_centers))) - used_columns
            for col in unused_columns:
                object_id = object_ids[col]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.center_points[object_id]
                    del self.disappeared[object_id]

            # Add new objects that were not matched with existing ones
            unused_rows = set(range(len(input_centers))) - used_rows
            for row in unused_rows:
                cx, cy, rect = input_centers[row]
                self.center_points[self.id_count] = (cx, cy)
                self.disappeared[self.id_count] = 0
                objects_bbs_ids.append([rect[0], rect[1], rect[2], rect[3], self.id_count])
                self.id_count += 1

        return objects_bbs_ids

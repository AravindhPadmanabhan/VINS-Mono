from cv_bridge import CvBridge
from tapnet.tapnext.tapnext_online import TAPNextOnline
import torch
import numpy as np
import cv2

class TAPNextTracker:
    def __init__(self, checkpoint, device='cuda'):
        self.model = TAPNextOnline(model_path=checkpoint, resolution=(256, 256), radius=8, threshold=0.5, use_certainty=True, device=device)
        self.device = device

        self.frame_no = -1

        self.max_queries = 100
        self.queries = torch.zeros(1,0,3).to(self.device)
        self.new_queries = torch.zeros(1,0,3).to(self.device)
        self.removed_indices = None
        self.cur_tracks = None

        self.is_first_step = True
        self.init_img = None
        self.latest_img = None
        self.reset_interval = 48
        self.delay = 0

    def update_queries(self, new_points, removed_indices): 
        init_queries = self.queries.shape[1]
        self.removed_indices = removed_indices
        if len(new_points) > 0:
            new_points = torch.tensor(new_points, dtype=torch.float32).to(self.device)  # Shape: (N,2)
            frame = torch.ones(new_points.shape[0], 1).to(self.device) * (self.frame_no - self.delay)  # Shape: (N,1)
            self.new_queries = torch.cat((frame, new_points), dim=1).unsqueeze(0)  # Shape: (1,N,3)
        else:
            self.new_queries = torch.zeros(1,0,3).to(self.device)

        if self.queries.shape[1] == 0:
            self.queries = self.new_queries
            return
        
        if len(removed_indices) > 0:
            mask = torch.ones(self.queries.shape[1], dtype=torch.bool)  # Create a mask for all points
            mask[removed_indices] = False
            self.queries = self.queries[:, mask, :]
        
        self.queries = torch.cat((self.queries, self.new_queries), dim=1)
        assert init_queries + len(new_points) - len(removed_indices) == self.queries.shape[1], "Queries update failed"
        
    def init_image(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1,H,W,3)
        self.frame_no += 1
        self.init_img = img_tensor

    def model_reset(self):
        # Tracking till frame T is done and frame T+1 is here at this point. Model restart would be with T, T+1 and the updated queries in T
        self.model.reset()
        self.delay += self.reset_interval

        mask = torch.ones(self.cur_tracks.shape[0], dtype=torch.bool)
        mask[self.removed_indices] = False
        query_coords = self.cur_tracks[mask]  # Get the coordinates of the remaining tracks
        query_coords = torch.cat((query_coords, self.new_queries[0, :, 1:3]), dim=0) 
        frame = torch.ones(query_coords.shape[0], 1).to(self.device) * (self.frame_no - self.delay)  # Shape: (N,1)
        self.queries = torch.cat((frame, query_coords), dim=1).unsqueeze(0)  # Shape: (1,N,3)

        __ = self.model(frame=self.latest_img, queries=self.queries)
        self.removed_indices = []


    def track(self, image):
        if self.frame_no % self.reset_interval == 0 and self.frame_no > 0:
            self.model_reset()

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).unsqueeze(0).to(self.device)  # Shape: (1,H,W,3)
        self.frame_no += 1
        self.latest_img = img_tensor

        assert img_tensor.shape[3] == 3

        if self.is_first_step:
            if self.queries.shape[1] == 0:
                self.init_img = img_tensor
                self.frame_no = 0
                print("No queries provided, skipping tracking.")
                return None, None
            __ = self.model(frame=self.init_img, queries=self.queries)
            self.is_first_step = False

        tracks, status = self.model(frame=img_tensor, queries=self.queries, removed_indices=self.removed_indices)

        self.cur_tracks = tracks[0,0]
        return tracks[0,0], status[0,0]
    
    def debug_tracks(self):
        latest_frame_np = self.latest_img.squeeze(0).cpu().numpy()
        latest_frame_np = latest_frame_np.astype(np.uint8)
        latest_frame_np = cv2.cvtColor(latest_frame_np, cv2.COLOR_RGB2BGR)
        if self.cur_tracks is not None:
            for i in range(self.queries.shape[1]):
                x, y = int(self.cur_tracks[i,0]), int(self.cur_tracks[i,1])
                if i < self.queries.shape[1] - self.new_queries.shape[1]:
                    cv2.circle(latest_frame_np, (x, y), 5, (0, 255, 0), -1)
                else:
                    cv2.circle(latest_frame_np, (x, y), 5, (255, 0, 0), -1)

        return latest_frame_np
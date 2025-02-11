from cv_bridge import CvBridge
from cotracker.predictor import CoTrackerPredictor
import torch
import numpy as np
import cv2

class CoTrackerWindow:
    def __init__(self, checkpoint, device='cuda'):
        self.model = CoTrackerPredictor(checkpoint=checkpoint)
        self.model.to(device)
        self.video = []
        self.frame_numbers = []
        self.frame_no = 1
        self.video_len = 24
        # self.video_padded = None
        self.max_queries = 100
        self.queries = None
        self.cur_tracks = None
        # self.track_status = torch.ones(self.max_queries, dtype=torch.bool).to(device)
        self.new_queries = None
        self.device = device

    def add_image(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1)
        # print(img_tensor)
        self.video.append(img_tensor)
        self.frame_numbers.append(self.frame_no)
        if len(self.video) == 1:
            self.video.extend([self.video[0]] * (self.video_len - 1))
            self.frame_numbers.extend([self.frame_numbers[0]] * (self.video_len - 1))
        if len(self.video) > self.video_len:
            self.video = self.video[-self.video_len:]
            self.frame_numbers = self.frame_numbers[-self.video_len:]
        self.frame_no += 1

    def update_queries(self, new_points, removed_indices):
        if len(removed_indices) > 0:
            new_points = torch.tensor(new_points, dtype=torch.float32).to(self.device)  # Shape: (N,2)
            frame = torch.ones(new_points.shape[0], 1).to(self.device) * (self.video_len - 1)  # Shape: (N,1)
            self.new_queries = torch.cat((frame, new_points), dim=1).unsqueeze(0)  # Shape: (1,N,3)
            if self.queries is None:
                self.queries = self.new_queries
                self.max_queries = self.queries.shape[1]
            else:
                mask = torch.ones(self.queries.shape[1], dtype=torch.bool)  # Create a mask for all points
                mask[removed_indices] = False
                self.queries = self.queries[:, mask, :]
                self.queries = torch.cat((self.queries, self.new_queries), dim=1)
        else:
            self.new_queries = torch.zeros(1,0,3).to(self.device)
        
        # Move queries behind by one frame:
        self.queries[0, :, 0] -= 1
        if self.cur_tracks is not None:
            out_of_window_mask = self.queries[0, :, 0] < 0
            traces = torch.cat((torch.zeros(1,self.max_queries,1).to(self.device), self.cur_tracks[:,1,:,:]), dim=-1)
            self.queries = torch.where(out_of_window_mask.unsqueeze(-1), traces, self.queries)

    def track(self):
        if self.queries is None:
            return None, None

        window_frames = torch.stack(self.video).to(self.device).unsqueeze(0)
        assert (window_frames.shape[1] == self.video_len) and (window_frames.shape[2] == 3), "Input video length does not match required length"
        tracks, vis = self.model(window_frames, queries=self.queries, backward_tracking=True)

        # self.track_status = vis
        self.cur_tracks = tracks
        print("vis shape: ", vis.shape)

        return tracks[0,-1,:,:], vis[0,-1]
    
    def debug_tracks(self):
        latest_frame = self.video[-2]
        latest_frame_np = latest_frame.permute(1, 2, 0).cpu().numpy()
        latest_frame_np = latest_frame_np.astype(np.uint8)
        latest_frame_np = cv2.cvtColor(latest_frame_np, cv2.COLOR_RGB2BGR)

        if self.cur_tracks is not None:
            for i in range(self.max_queries):
                x, y = int(self.cur_tracks[0,-2,i,0]), int(self.cur_tracks[0,-2,i,1])
                if i < self.max_queries - self.new_queries.shape[1]:
                    cv2.circle(latest_frame_np, (x, y), 5, (0, 255, 0), -1)
                else:
                    cv2.circle(latest_frame_np, (x, y), 5, (255, 0, 0), -1)

        return latest_frame_np

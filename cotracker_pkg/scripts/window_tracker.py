from cv_bridge import CvBridge
from cotracker.predictor import CoTrackerOnlinePredictor
import torch
import numpy as np
import cv2

class CoTrackerWindow:
    def __init__(self, checkpoint, device='cuda'):
        self.model = CoTrackerOnlinePredictor(checkpoint=checkpoint)
        self.model.to(device)
        self.video = []
        self.video_len = self.model.model.window_len + self.model.step
        # self.video_padded = None
        self.max_queries = 100
        self.queries = None
        self.cur_tracks = None
        self.track_status = torch.ones(self.max_queries, dtype=torch.bool).to(device)
        self.new_queries = None
        self.device = device

    def add_image(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1)
        # print(img_tensor)
        self.video.append(img_tensor)
        if len(self.video) == 1:
            self.video = self.video + [self.video[-1]] * (self.video_len - len(self.video))
        if len(self.video) > self.video_len:
            self.video.pop(0)

    def get_queries(self, num_queries):
        query_frame = self.video[-1]

        image_np = query_frame.permute(1, 2, 0).cpu().numpy()
        image_np = image_np.astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

        sift = cv2.SIFT_create(nfeatures=num_queries)
        keypoints = sift.detect(gray_image, None)
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)  # Sort by response

        keypoints = keypoints[:num_queries]

        # Convert keypoints to a list of (x, y) coordinates
        keypoints_xy = [kp.pt for kp in keypoints]  # Extract x, y coordinates

        # Convert to PyTorch tensors
        keypoints_tensor = torch.tensor(keypoints_xy, dtype=torch.float32).to(self.device)  # Shape: (N, 2)
        
        indices = torch.ones(num_queries, 1).to(self.device) * (self.video_len - 1)
        self.queries = torch.cat((indices, keypoints_tensor), dim=1)  # Shape: (N, 3)
        self.queries = self.queries.unsqueeze(0) # Shape: (1, N, 3)

    def update_queries(self, tracks):
        tracked_queries = self.queries[:, self.track_status, :]
        
        if self.max_queries - tracked_queries.shape[1] > 0:
            self.new_queries = self.get_queries(self.max_queries - tracked_queries.shape[1])
            self.queries = torch.cat((tracked_queries, self.new_queries), dim=1)

        # Move queries behind by one frame:
        self.queries[0, :, 0] -= 1
        out_of_window_mask = self.queries[0, :, 0] < 0
        traces = torch.cat((torch.zeros(1,100,1), tracks[:,1,:,:]), dim=-1)
        self.queries = torch.where(out_of_window_mask, traces, self.queries)

    def track(self):
        if self.queries is None:
            self.get_queries(self.max_queries)

        is_first_step = True
        for i in range(3):
            video_chunk = self.video[:(i+1)*self.model.step]
            window_frames = video_chunk[-self.model.model.window_len:]
            window_frames = torch.stack(window_frames).to(self.device).unsqueeze(0)
            tracks, vis, conf = self.model(window_frames, is_first_step=is_first_step, queries=self.queries)
            if is_first_step:
                is_first_step = False

        self.track_status = ((conf[0,-1,:] > 0.6) * vis[0,-1,:]) == 1
        self.cur_tracks = tracks

        self.update_queries(tracks)

        return tracks, vis
    
    def debug_features(self):
        # Mark the queries on the image
        query_frame = self.video[0]
        query_frame_np = query_frame.permute(1, 2, 0).cpu().numpy()
        query_frame_np = query_frame_np.astype(np.uint8)
        query_frame_np = cv2.cvtColor(query_frame_np, cv2.COLOR_RGB2BGR)
        for query in self.queries[0]:
            x, y = int(query[1]), int(query[2])
            cv2.circle(query_frame_np, (x, y), 5, (0, 255, 0), -1)

        return query_frame_np
    
    def debug_tracks(self):
        latest_frame = self.video[-1]
        latest_frame_np = latest_frame.permute(1, 2, 0).cpu().numpy()
        latest_frame_np = latest_frame_np.astype(np.uint8)
        latest_frame_np = cv2.cvtColor(latest_frame_np, cv2.COLOR_RGB2BGR)

        for i in range(self.max_queries):
            x, y = int(self.cur_tracks[0,-1,i,0]), int(self.cur_tracks[0,-1,i,1])
            if self.track_status[i]:
                cv2.circle(latest_frame_np, (x, y), 5, (0, 255, 0), -1)

        for query in self.new_queries[0]:
            x, y = int(query[1]), int(query[2])
            cv2.circle(latest_frame_np, (x, y), 5, (0, 0, 255), -1)

        return latest_frame_np


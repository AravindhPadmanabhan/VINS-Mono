from cv_bridge import CvBridge
from trackon.track_on_ff import TrackOnFF
from trackon.track_on_cfg import TrackOnCfg
import torch
import numpy as np
import cv2

from load_checkpoint import restart_from_checkpoint_not_dist

class TrackOnTracker:
    def __init__(self, checkpoint, device='cuda'):
        args = TrackOnCfg(checkpoint_path=checkpoint)
        self.model = TrackOnFF(args)
        restart_from_checkpoint_not_dist(args, run_variables={}, model=self.model)
        self.model.to(device).eval()
        self.model.set_memory_size(args.val_memory_size, args.val_memory_size)
        self.model.visibility_treshold = args.val_vis_delta
        self.model.confidence_treshold = 0.8
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
        
    def update_queries(self, new_points, removed_indices): 
        init_queries = self.queries.shape[1]
        self.removed_indices = removed_indices
        if len(new_points) > 0:
            new_points = torch.tensor(new_points, dtype=torch.float32).to(self.device)  # Shape: (N,2)
            frame = torch.ones(new_points.shape[0], 1).to(self.device) * self.frame_no  # Shape: (N,1)
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
        # if self.queries.shape[1] == 0:
        #     return None, None

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)  # Shape: (1,3,H,W)
        self.frame_no += 1
        self.init_img = img_tensor

    def track(self, image):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)  # Shape: (1,3,H,W)
        self.frame_no += 1
        self.latest_img = img_tensor

        assert img_tensor.shape[1] == 3

        with torch.no_grad():
            if self.is_first_step:
                if self.queries.shape[1] == 0:
                    self.init_img = img_tensor
                    self.frame_no = 0
                    # self.frame_no -= 1
                    # self.is_first_step = True
                    print("No queries provided, skipping tracking.")
                    return None, None
                self.model.init_queries_and_memory(self.queries.squeeze(0), self.init_img)
                __ = self.model.ff_forward(self.init_img)
                self.model.update_queries_and_memory(self.queries.squeeze(0), img_tensor, self.removed_indices)
                self.is_first_step = False
            else:
                self.model.update_queries_and_memory(self.queries.squeeze(0), img_tensor, self.removed_indices)
            tracks, vis, conf = self.model.ff_forward(img_tensor)

            # self.track_status = vis
        self.cur_tracks = tracks
        return tracks, conf
    
    def debug_tracks(self):
        latest_frame_np = self.latest_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
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
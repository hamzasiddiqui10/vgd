## VLM Shellgame multiview

- We want multiview videos such that:
  - videos test **correspondence** (unique object identification) AND **geometric understanding**   
  (cam position understanding) on the shell game.
  - One view should not be enough to predict.
  - Different cam positions not just two cams across from each other
  - single-view shellgame vids were 9-10s long. These multiview ones are slightly longer at 11-12s


- VLM is **gemini-3.8-flash (processing mode: 'agentic')** (model performed very well in single-view shellgame)  
→ A,B,C labeled on table, visible in cam1 (POV). At start, show where puck is, move camera to other view cam1,   
shuffle cups, move back into cam0 view and place on A B and C markers.
- 3 types of videos:
  - shellgame_mv_a – cam0 and cam1 look in same direction
  - shellgame_mv_b – cam0 and cam1 across from eachother (lateral inversion)
  - shellgame_mv_c – cam0 same and cam1 on some diagonal angle or even left or right view camera.
- For each video model picks from A,B,C


### Workflow for multiview eval
Compress cam0 and cam1 down to 480p-20fps --> Stitch both views together --> Write custom prompt --> API call

## Result:
- Not all videos are processing. 41 total videos and only (a+b+c = 16+8+3 = 27) vids processed. Rest threw timeout error even after more than 5 attempts per video.
- **top1acc shellgame_mv_a (0.75) > shellgame_mv_b (0.38) > shellgame_mv_c (0.0)** -- as expected
- CONCLUSION: Stitching together information from the 2 perspectives is something the model CANNOT do as evident from very poor performance on shellgame_mv_b and shellgame_mv_c.
- Close error analysis shellgame_mv_b showed 2 types of errors: 
  - A. In some videos model is follwoing the puck for a few moves correctly and then just abruptly returning result when more swaps are still happening (e.g. 261125-112001). 
  - B. Proper mistakes on model's part in tracking swaps (e.g. 261125-112139).


- no pattern in prompt/completion tokens.
- No real pattern in nframes seen by model. avg 100-120 frames_seen across vids. Highest is 205 frames (shellgame_mv_a: 261125-104759) still wrong prediction. swap sequence suggests the model just did not see all swaps because it was correct for first few swaps then just returned answer.
- **camera_setup_reasoning:** I asked in the prompt for the model to comment on what it thinks the camera setup is like or how it thinks cam0 is positioned relative to cam1. (I disabled it for quicker evals after 18 videos because it was enough)
  - On 12/13 shellgame_mv_a videos the reasoning is good. This was simplest setup.
  - On 2/3 shellgame_mv_b videos the reasoning is good. 
  - On 3/3 shellgame_mv_c videos reasoning is good.
  - CONCLUSION: I think geometric understanding of model is fairly decent in 2-camera setup insofar as knowing how the cameras are positioned. 



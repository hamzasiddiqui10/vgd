## VLM multiview which_shape_next_cam0

- VLM is **gemini-3.8-flash (processing mode: 'agentic')** 
- I made a 4-sided rectangular paper prism with a unique shape on each side. 
- Cam0 is birds eye only sees object top AND cam1 POV view but level with object on table so it can see only one side at a time.
10 vids where i just rotate the cube consistently and ask the model to tell me at end of video which shape it thinks will appear on cam0 with one more turn?
  - which_shape_next_cam0**_a** – 11 vids rotating clockwise. So CURRENT SHAPE on cam1 will be NEXT SHAPE ON cam0. If fewer than 4 rotations made, model cannot predict next shape using cam0 alone, it needs cam1 as well.
  - which_shape_next_cam0**_b** – 5 vids rotating anti-clockwise. So current shape on cam1 doesnt help. In _b, I will rotate for longer than 4 turns so cam0 itself will recognize repeating pattern and maybe predict. *[probably wont use at all as they are useless with multi-view (realised while recording)]*
- For each video model picks from ['square','rectangle','circle','triangle'] as next shape on cam0.
- Caveat: This experiment not designed as meticulously like the shellgame-multiview. 

### Workflow for multiview eval
Compress cam0 and cam1 down to 480p-20fps --> Stitch both views together --> Write custom prompt --> API call

## Result:
- All api calls ran smoothly, quickly and with much fewer tokens (granted the videos are short at ~5s)
- Perfect prediction on ALL 16 videos (both _a and _b experiments). **top1acc = 1.0**
- 11vids of _a that were correct show that model understands whats is happening. (e.g. 261125-130240 has only 3 turns)
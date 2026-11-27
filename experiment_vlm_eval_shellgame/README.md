## VLM Shellgame

- VLM is **gemini-3.8-flash** (has a better video enc)  
(Theoretically should outperform the glm model -- OPENROUTER: **gemini-3.8-flash**)
- Videos: A,B,C labeled on the table with 3 inverted cups. Each video starts by showing where the puck is then covers it with a cup.  
then cups are moved around in shell game style.
- evaluated on single videos of shell game in POV.
- 3 types of videos: slow, medium, fast to see temporal resoning of the model
- For each video model picks from A,B,C


## Result:
Results for the gemini-3.8-flash (processing mode: 'agentic').

Very solid results. In the results table you will find not just top1acc by fast-medium-slow but also tokens and $$cost.

Main Takeaway: The model can track the medium and slow movements very well which is evident from the swap_sequence i ask it to provide (i have manually checked examples). This is tru for half of fast videos as well.

Big Idea of agentic processing: I learned on Friday that by default the way frame sampling was done for videos given to gemini models was 1 frame per second. That is why the performance initially was poor. Now with the "agentic" processing the model is dynamically deciding how many frames to pick for every api call and the performance is great.

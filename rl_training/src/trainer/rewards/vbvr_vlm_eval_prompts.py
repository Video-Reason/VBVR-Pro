"""
Task-specific VLM-judge prompts for the 100 VBVR-Bench tasks.

Generated from the evaluator source: for each task the aspects are exactly the
sub-scores its evaluator computes, and each aspect's bullets describe what that
sub-score actually measures. Thresholds are deliberately omitted -- the judge
applies its own standard to the same criteria.

Weights are NOT shown to the judge. The judge reports its own weights, and the
scorer's true weights live in dim_map.py, so the two can be compared.

Do not hand-edit: regenerate with `python gemini_eval/prompt_spec/gen.py`.
"""

EVAL_PROMPTS = {

'G-3_stable_sort_data-generator': """
You are evaluating a generated video for a stable-sort task.

Task: The scene contains two types of shapes, each type has three shapes of different sizes arranged randomly. Keep all shapes unchanged in appearance (type, size, and color). Only rearrange their positions: first group the shapes by type, then within each group, sort the shapes from smallest to largest (left to right), and arrange all shapes in a single horizontal line from left to right.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Arrangement: Whether the objects end up sorted into the intended layout.
   - horizontal alignment: the objects sit on a common horizontal line
   - grouping: objects of the same shape are grouped together
   - within-group order: inside each shape group the objects run in order of size

2. Foreground Consistency: Whether the objects themselves survive the rearrangement.
   - every object present at the start is still present at the end with unchanged appearance
   - no extra or duplicated objects have appeared

3. Background Consistency: Whether the area outside the objects stays the plain original background.
   - the non-object area remains uniform, with no leftover marks or drag artifacts

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
arrangement_score: <0-100>
fore_consistency_score: <0-100>
back_consistency_score: <0-100>
arrangement_weight: <0-100>
fore_consistency_weight: <0-100>
back_consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-13_grid_number_sequence_data-generator': """
You are evaluating a generated video for a grid numbered-waypoint path task.

Task: The scene shows a 10x10 grid with a green start point, a red end point, and yellow cells marked with numbers 1, 2, and 3. An orange circular agent is positioned at the green start point. The agent can move to adjacent cells (up, down, left, right). Starting from the green start point, the agent must visit the numbered yellow cells in numerical order (1, then 2, then 3), taking the shortest path between each consecutive pair of numbered cells. After visiting all numbered cells in sequence, the agent must reach the red end point, also following the shortest path.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Segment Path Accuracy: Whether the agent walks each leg of the route (start to waypoint 1, then on to the next, finally to the end) along a shortest path, taking the numbered waypoints in ascending order.
   - on each leg the agent stays on cells belonging to a shortest route for that leg
   - the agent actually traverses each leg all the way to its waypoint
   - if a waypoint is never reached, every later leg is worthless

2. Motion Continuity: Whether the agent moves as one continuous object across the video.
   - no teleporting between distant cells, no vanishing and reappearing, no flicker

3. Background Preservation: Whether everything other than the moving agent stays exactly as in the input frame.
   - grid lines, numbers, and cell colours are unchanged wherever the agent is not standing

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
mean_segment_score_score: <0-100>
continuity_factor_score: <0-100>
bg_preservation_score: <0-100>
mean_segment_score_weight: <0-100>
continuity_factor_weight: <0-100>
bg_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-15_grid_avoid_obstacles_data-generator': """
You are evaluating a generated video for a grid obstacle-avoiding path task.

Task: The scene shows a 10x10 grid with a blue start square (containing a yellow circular agent), a red end square, and multiple black X marks indicating obstacles. Starting from the blue start square, the agent can move to adjacent cells (up, down, left, right) each step. The goal is to move the agent to the red end square along the shortest path without entering any cells marked with black X obstacles.

You will receive: the first frame (input) and the generated video.

Score the following 5 aspects (each 0-100):

1. Path Correctness: Whether the agent stays on cells belonging to a valid shortest route from start to goal.
   - at every frame the agent sits on (or very near) a cell of an optimal obstacle-free route

2. Path Completion: Whether the agent traverses the whole route rather than stopping partway.
   - it visits the route's cells from the start cell through to the goal cell

3. Motion Continuity: Whether the agent moves as one continuous object.
   - no teleporting between distant cells, no vanishing and reappearing, no flicker

4. Obstacle Avoidance: Whether the agent ever steps onto an obstacle cell.
   - every distinct obstacle cell the agent touches is a serious violation

5. Background Preservation: Whether everything other than the moving agent stays as in the input frame.
   - grid lines, obstacles, start and goal markers are unchanged wherever the agent is not standing

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
proximity_score: <0-100>
coverage_score: <0-100>
continuity_factor_score: <0-100>
obstacle_multiplier_score: <0-100>
bg_preservation_score: <0-100>
proximity_weight: <0-100>
coverage_weight: <0-100>
continuity_factor_weight: <0-100>
obstacle_multiplier_weight: <0-100>
bg_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-16_grid_go_through_block_data-generator': """
You are evaluating a generated video for a grid must-visit-blocks path task.

Task: The scene shows a 10x10 grid with a green start square (containing an orange circular agent), a red end square, and multiple blue rectangular blocks. Starting from the green start square, the agent can move to adjacent cells (up, down, left, right) each step. The goal is to move the agent to the red end square along the shortest path that passes through all blue blocks (the agent must visit every blue block before reaching the red end square).

You will receive: the first frame (input) and the generated video.

Score the following 5 aspects (each 0-100):

1. Path Correctness: Whether the agent stays on cells belonging to a shortest route that collects every required block before reaching the goal.
   - at every frame the agent sits on (or very near) a cell of such a route

2. Path Completion: Whether the agent traverses the whole route rather than stopping partway.
   - it progresses from the start cell through to the goal cell

3. Motion Continuity: Whether the agent moves as one continuous object.
   - no teleporting between distant cells, no vanishing and reappearing, no flicker

4. Required Blocks Visited: Whether the agent actually steps on every block it must pass through.
   - each required block that is never visited is a serious violation

5. Background Preservation: Whether everything other than the moving agent stays as in the input frame.
   - grid lines, the blocks, start and goal markers are unchanged wherever the agent is not standing

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
proximity_score: <0-100>
coverage_score: <0-100>
continuity_factor_score: <0-100>
block_multiplier_score: <0-100>
bg_preservation_score: <0-100>
proximity_weight: <0-100>
coverage_weight: <0-100>
continuity_factor_weight: <0-100>
block_multiplier_weight: <0-100>
bg_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-18_grid_shortest_path_data-generator': """
You are evaluating a generated video for a grid shortest-path task.

Task: The scene shows a 10x10 grid with a green start square (containing a purple circular agent) and a pink end square. Starting from the green start square, the agent can move to adjacent cells (up, down, left, right) each step. The goal is to move the agent to the pink end square along the shortest path.

You will receive: the first frame (input) and the generated video.

Score the following 4 aspects (each 0-100):

1. Path Correctness: Whether the agent stays on cells belonging to a shortest route from the start cell to the goal cell.
   - at every frame it sits on (or very near) a cell of a shortest route
   - detours off the shortest route lower this

2. Path Completion: Whether the agent traverses the whole route rather than stopping partway.
   - it progresses from the start cell through to the goal cell

3. Motion Continuity: Whether the agent moves as one continuous object.
   - it advances cell by cell, without teleporting, vanishing, or flickering

4. Background Preservation: Whether everything other than the moving agent stays as in the input frame.
   - grid lines, start and goal cells are unchanged wherever the agent is not standing

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
proximity_score: <0-100>
coverage_score: <0-100>
continuity_factor_score: <0-100>
bg_preservation_score: <0-100>
proximity_weight: <0-100>
coverage_weight: <0-100>
continuity_factor_weight: <0-100>
bg_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-21_multiple_occlusions_vertical_data-generator': """
You are evaluating a generated video for a vertical occlusion sweep task.

Task: The scene shows 3 objects arranged in a horizontal line in the center of the frame, with a dark rectangular mask initially positioned above them. Move the mask vertically downward in a continuous motion until it leaves the frame. As it moves, the mask passes in front of the objects, temporarily blocking them from view.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Mask Path Validity: Whether the dark mask sweeps downward across the scene and off the bottom.
   - it travels downward and never reverses upward
   - it continues until it leaves the bottom of the frame
   - its shape, size, and colour stay the same while it travels

2. Occlusion Correctness: Whether the mask genuinely hides the objects while passing over them.
   - while the mask overlaps an object, that object is actually covered rather than showing through

3. Elements Preservation: Whether the scene is intact where the mask is not currently covering it.
   - the coloured objects keep their positions and appearance, and reappear unchanged after the mask passes
   - the area outside the objects stays unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
mask_path_vadility_score: <0-100>
occlusion_correctness_score: <0-100>
elements_preservation_score: <0-100>
mask_path_vadility_weight: <0-100>
occlusion_correctness_weight: <0-100>
elements_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-25_seperate_object_spinning_data-generator': """
You are evaluating a generated video for a separate objects while spinning task.

Task: The scene shows 3 objects on the left side and dashed target outlines on the right side. The dashed target outlines remain completely stationary. For each object, first rotate it in place to match the orientation of its corresponding dashed target outline, then move it horizontally to the right so that it aligns exactly with and fits within its corresponding dashed target outline.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Alignment: Whether each object ends up at the target pose it belongs to.
   - each object's final position and orientation match its intended target

2. Motion Quality: Whether the objects get there with clean, well-ordered motion.
   - each object travels smoothly and stays visible throughout
   - no teleporting between distant positions
   - the objects separate in a sensible order rather than all at once
   - no extra or hallucinated objects appear in the final frame
   - the background stays unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
alignment_score: <0-100>
non_alignment_score_score: <0-100>
alignment_weight: <0-100>
non_alignment_score_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-29_chart_extreme_with_data_data-generator': """
You are evaluating a generated video for a chart extreme-value marking task.

Task: The scene shows a bar chart titled 'Monthly Product Sales Statistics 2024' with Month on the x-axis and Sales on the y-axis. Find the month with the lowest sales and draw a red rectangular border around the corresponding bar to highlight it.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Marking Accuracy: Whether the red marks land on the correct extreme bar(s) or point(s) of the chart.
   - each expected red mark exists and traces the region it should mark
   - the number of red marks is right: missing marks and extra marks both count against this

2. Background Consistency: Whether the white background of the chart is left untouched.
   - the white canvas outside the chart's ink stays clean

3. Foreground Consistency: Whether the chart itself — everything that is not the red marking — is unchanged.
   - bars, axes, and labels keep their shape, position, and colour

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
accuracy_score: <0-100>
back_consistency_score: <0-100>
fore_consistency_score: <0-100>
accuracy_weight: <0-100>
back_consistency_weight: <0-100>
fore_consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-31_directed_graph_navigation_data-generator': """
You are evaluating a generated video for a directed graph navigation task.

Task: The scene shows a network of nodes connected by directed edges (edges with arrows indicating direction) with a green starting node, a red ending node, and a blue triangular agent positioned at the green starting node. The agent can only move along edges in the direction they point (from the source node to the target node, cannot move backwards), moving from one node to an adjacent node each step. Move the blue triangular agent from the green starting node to the red ending node along the path with the minimum number of steps.

You will receive: the first frame (input) and the generated video.

Score the following 4 aspects (each 0-100):

1. Completion: Whether the blue triangular agent ends the video at the correct destination node.
   - its final position coincides with the goal node
   - exactly one agent is present; extra agents count against this

2. Path Validity: Whether the route the agent traces matches the correct route through the graph.
   - the trajectory it sweeps out follows the correct edges rather than cutting across the canvas
   - it advances continuously, without large teleporting jumps
   - exactly one agent is present throughout

3. Foreground Preservation: Whether the graph itself is left untouched while the agent moves.
   - nodes and edges keep their positions, colours, and sizes

4. Background Preservation: Whether the blank canvas around the graph stays clean.
   - the white area outside nodes and edges is unchanged, with no smearing or stray marks

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
path_validity_score: <0-100>
foreground_preservation_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
path_validity_weight: <0-100>
foreground_preservation_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-39_attention_shift_different_data-generator': """
You are evaluating a generated video for an attention-shift task.

Task: The scene shows two objects, one on the left and one on the right, with a green attention box around the left object. The objects remain stationary and unchanged throughout. Move the green attention box from the left object to the right object.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Box Position: Whether the final frame's green hollow box sits on the correct target object.
   - the box is centred on the target object — the OTHER object, not the one it started on
   - the box encloses that object without being drastically oversized
   - exactly one green hollow box is present

2. Consistency: Whether the final frame is otherwise identical to the input frame.
   - the two objects themselves are unchanged where they sit
   - the plain background outside the objects and the box is unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
box_position_score: <0-100>
consistency_score: <0-100>
box_position_weight: <0-100>
consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-41_grid_highest_cost_data-generator': """
You are evaluating a generated video for a highest-cost grid path task.

Task: The scene shows a 4x4 grid with cost values displayed in each cell, a green start cell (containing a yellow Pac-Man agent) at the top-left, and a red goal cell at the bottom-right. Starting from the green start cell, the agent can move to adjacent cells (up, down, left, right) one step at a time. Move the agent to the red goal cell along the path with the highest total cost (the sum of all cost values of cells visited along the path), showing the complete movement process step by step.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Cost Optimality: Whether the route the agent walks collects as much total cost as the best possible route.
   - the cells it steps through carry high values, matching the highest-value route from the start cell to the bottom-right end cell

2. Move Legality: Whether every move the agent makes is a legal one.
   - each step goes to an orthogonally adjacent cell — no diagonal steps, no jumps
   - no cell is revisited
   - the agent stops once it reaches the end cell instead of wandering on

3. Background Preservation: Whether everything other than the moving agent stays as in the input frame.
   - the grid lines and the printed cost numbers are unchanged wherever the agent is not standing

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
cost_optimality_score: <0-100>
move_legality_score: <0-100>
bg_preservation_score: <0-100>
cost_optimality_weight: <0-100>
move_legality_weight: <0-100>
bg_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-43_understand_scene_structure_data-generator': """
You are evaluating a generated video for a scene-structure outlining task.

Task: The scene shows a floorplan with multiple rooms. Identify the kitchen and draw a green rectangular box around it to highlight it.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Outlining Accuracy: Whether the green outlines land on the correct structural regions of the scene.
   - each region that should be outlined is traced by a green outline
   - the number of outlines is right: missing and extra outlines both count against this

2. Background Consistency: Whether the white background stays untouched.
   - the white canvas outside the drawing stays clean

3. Foreground Consistency: Whether the original scene content is unchanged.
   - walls, furniture, and all pre-existing lines keep their shape, position, and colour

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
accuracy_score: <0-100>
back_consistency_score: <0-100>
fore_consistency_score: <0-100>
accuracy_weight: <0-100>
back_consistency_weight: <0-100>
fore_consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-45_key_door_matching_data-generator': """
You are evaluating a generated video for a key-door maze task.

Task: The scene shows a maze with a green circular agent, colored diamond-shaped keys, and colored hollow rectangular doors. Find the Blue key and then navigate to the matching Blue door, showing the complete movement process step by step.

You will receive: the first frame (input) and the generated video.

Score the following 5 aspects (each 0-100):

1. Path Correctness: Whether the agent stays on the corridor cells of the correct route from its start to the key that opens the door.
   - at every frame it sits on a cell of that route rather than wandering elsewhere

2. Path Completion: Whether the agent walks the whole route rather than stopping partway.
   - it progresses from the start cell all the way to the target key

3. Motion Continuity: Whether the agent moves as one continuous object.
   - it advances cell by cell, without teleporting, vanishing, or flickering

4. Correct Key Reached: Whether the agent reaches the key whose colour matches the door it must open.
   - it visits the correct-colour key; going to a wrong-colour key or to no key at all is a serious failure

5. Wall Avoidance: Whether the agent ever steps into a wall cell.
   - every distinct wall cell it intrudes into is a serious violation

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
proximity_score: <0-100>
coverage_score: <0-100>
continuity_factor_score: <0-100>
key_multiplier_score: <0-100>
wall_multiplier_score: <0-100>
proximity_weight: <0-100>
coverage_weight: <0-100>
continuity_factor_weight: <0-100>
key_multiplier_weight: <0-100>
wall_multiplier_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-51_predict_next_color_data-generator': """
You are evaluating a generated video for a predict next colour task.

Task: Predict the next color in the sequence.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Completion: Whether the empty slot is filled with the colour that correctly continues the pattern.
   - the newly drawn block occupies the empty slot and shows the colour the pattern demands

2. Foreground Preservation: Whether the blocks that were already there are unchanged.
   - the existing coloured blocks keep their colours, positions, and sizes

3. Background Preservation: Whether the canvas around the blocks stays clean.
   - the area outside the blocks is unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
foreground_preservation_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
foreground_preservation_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-131_select_next_figure_increasing_size_sequence_data-generator': """
You are evaluating a generated video for a circle-the-next-figure task.

Task: A sequence of shapes arranged from smallest to largest. Circle the next shape in the candidate area that continues this pattern.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the existing shapes keep their type, size, and position
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle marks the candidate that continues the small-to-large size progression.
   - the circle encloses the correct candidate
   - no wrong candidate is circled — circling a wrong one subtracts from the score
   - each circle unambiguously picks out one shape rather than straddling several

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-134_select_next_figure_large_small_alternating_sequence_data-generator': """
You are evaluating a generated video for a circle-the-next-figure task.

Task: A sequence of shapes arranged in a 'big-small-big' pattern. Circle the next shape in the candidate area that continues this 'big-small-big-small' pattern.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the existing shapes keep their type, size, and position
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle marks the candidate that continues the big-small-big-small alternation.
   - the circle encloses the correct candidate
   - no wrong candidate is circled — circling a wrong one subtracts from the score
   - each circle unambiguously picks out one shape rather than straddling several

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-138_spot_unique_non_repeated_color_data-generator': """
You are evaluating a generated video for a spot the unique colour task.

Task: The initial state displays multiple geometric shapes with different colors and shapes. Exactly one color is unique (appears only once), while other colors appear at least twice. The animation shows the process of identifying and marking the unique color shape with a black border.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Marking Accuracy: Whether the mark lands on the one shape whose colour appears only once.
   - the mark traces that shape and no other
   - exactly one shape is marked; a missing mark or extra marks both count against this

2. Background Consistency: Whether the white background stays untouched.
   - the white canvas outside the shapes stays clean

3. Foreground Consistency: Whether all the original shapes are unchanged.
   - every shape keeps its colour, position, and size; only the new mark is added

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
accuracy_score: <0-100>
back_consistency_score: <0-100>
fore_consistency_score: <0-100>
accuracy_weight: <0-100>
back_consistency_weight: <0-100>
fore_consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-158_identify_all_hollow_points_data-generator': """
You are evaluating a generated video for a circle every hollow point task.

Task: Find all the empty/outlined circles and mark them. Solid points are filled black, hollow points are white with black outline. Circle every hollow point you find.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circles changes.
   - the white background stays clean
   - every point keeps its type and its filled-vs-hollow appearance — a solid dot must not become hollow or vice versa
   - the red circles are thin annotations, not large red blobs

2. Selection Match: Whether every hollow (white, black-outlined) point is circled and no solid one is.
   - each hollow point is enclosed by a red circle
   - circling a solid point subtracts from the score, and doing so repeatedly is punished harder than missing one
   - each circle unambiguously picks out one point rather than straddling several

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-194_construct_concentric_ring_data-generator': """
You are evaluating a generated video for a concentric rings task.

Task: Create an animation where both circles travel from their initial separate positions to meet at the center, forming concentric circles. Both circles should have the same color.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Arrangement: Whether the rings end up concentric and centred.
   - all rings share a common centre
   - that shared centre sits at the centre of the frame

2. Ring Fidelity: Whether the set of rings is exactly the set that should be there.
   - every expected ring is present with the right radius
   - no extra or duplicated rings have been drawn

3. Background Consistency: Whether the area outside the rings stays plain white.
   - no stray marks or leftover ring fragments remain

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
arrangement_score: <0-100>
fore_consistency_score: <0-100>
back_consistency_score: <0-100>
arrangement_weight: <0-100>
fore_consistency_weight: <0-100>
back_consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-5_multi_object_placement_data-generator': """
You are evaluating a generated video for a multi-object placement task.

Task: The scene contains multiple colored objects and star markers. Keep all star markers unchanged in position. Move each colored object to the star marker with the same color using straight paths, aligning the center of each object with the center of its matching star marker.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Object Placement: Whether each coloured object ends up where it belongs, judged object by object and then averaged.
   - each object's final position matches its intended target
   - each object travels there directly rather than wandering
   - each object keeps its identity, colour, and size on the way
   - objects that appear out of nowhere drag this down

2. Stars Untouched: Whether the star markers stay exactly where they were.
   - the stars mark target positions and must not move; each star that moves is a serious violation

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
object_placement_score: <0-100>
star_penalty_score: <0-100>
object_placement_weight: <0-100>
star_penalty_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-8_track_object_movement_data-generator': """
You are evaluating a generated video for a track object movement task.

Task: The purple square marked with a green border is the only object that will move. It will move horizontally to align directly below the green rectangle marked with a red star. Track the movement with the green border as the object moves.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Movement: Whether the marked object (the one ringed by a green border in the first frame) actually travels horizontally.
   - it covers a substantial horizontal distance rather than staying put
   - it does not drift vertically
   - it keeps its shape and area while moving

2. Target Alignment: Whether the marked object ends up directly below the target marked by the red star.
   - its final horizontal position lines up with the target's horizontal position

3. Only One Moved: Whether every other object stays exactly where it was.
   - only the marked object may move; any other object that shifts is a violation

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
movement_score: <0-100>
target_alignment_score: <0-100>
only_one_moved_score: <0-100>
movement_weight: <0-100>
target_alignment_weight: <0-100>
only_one_moved_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-9_identify_objects_in_region_data-generator': """
You are evaluating a generated video for an identify objects in region task.

Task: Outline all trapezoids in the square region with a green border. Only outline objects within that region.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Marking Accuracy: Whether the green outlines land on exactly the objects lying inside the specified region.
   - every object inside the region is traced by a green outline
   - no object outside the region is outlined; missing and extra outlines both count against this

2. Background Consistency: Whether the white background stays untouched.
   - the white canvas outside the drawing stays clean

3. Foreground Consistency: Whether the original objects and the region boundary are unchanged.
   - everything that is not the new green outline keeps its shape, position, and colour

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
accuracy_score: <0-100>
back_consistency_score: <0-100>
fore_consistency_score: <0-100>
accuracy_weight: <0-100>
back_consistency_weight: <0-100>
fore_consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-10_shape_outline_fill_data-generator': """
You are evaluating a generated video for a visual analogy task.

Task: Animate the fill-to-outline transformation where the filled shape changes to outline style according to the established pattern. The question mark should smoothly transition to show the correct outline style.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Completion: Whether the bottom-right panel is filled with the shape the analogy demands: the same shape as the bottom-left one, with the same filled-vs-outline style change that relates the top two panels.
   - the drawn shape carries the right fill/outline style
   - its shape type, size, colour, and position within the panel are right

2. Foreground Preservation: Whether the panels that were already filled in are unchanged.
   - the shapes that were already there keep their sizes, colours, and positions

3. Background Preservation: Whether the canvas stays clean.
   - the canvas around the shapes stays clean and unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
foreground_preservation_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
foreground_preservation_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-12_shape_color_then_scale_data-generator': """
You are evaluating a generated video for a two-step colour-then-scale task.

Task: Complete the analogy by revealing the shape with the correct color and scale.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Completion: Whether both steps of the transformation are applied to the target shape.
   - the recoloured shape carries the colour the example demands
   - the rescaled shape has the size the example demands
   - each shape keeps its type and position

2. Foreground Preservation: Whether the example row is unchanged.
   - the shapes that were already there keep their sizes, colours, and positions

3. Background Preservation: Whether the canvas stays clean.
   - the canvas around the shapes stays clean and unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
foreground_preservation_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
foreground_preservation_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-13_shape_outline_then_move_data-generator': """
You are evaluating a generated video for a two-step outline-then-move task.

Task: Show the shape first becoming outline-only and then moving up or down. Both transformations should match the example pattern.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Completion: Whether both steps of the transformation are applied to the target shape.
   - the restyled shape switches between filled and outline as the example demands
   - the moved shape ends at the position the example demands
   - each shape keeps its type and size

2. Foreground Preservation: Whether the example row is unchanged.
   - the shapes that were already there keep their sizes, colours, and positions

3. Background Preservation: Whether the canvas stays clean.
   - the canvas around the shapes stays clean and unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
foreground_preservation_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
foreground_preservation_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-14_shape_scale_then_outline_data-generator': """
You are evaluating a generated video for a two-step scale-then-outline task.

Task: Complete the analogy by revealing the shape with the correct scale and outline style.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Completion: Whether both steps of the transformation are applied to the target shape.
   - the rescaled shape has the size the example demands
   - the restyled shape switches between filled and outline as the example demands
   - each shape keeps its type and position

2. Foreground Preservation: Whether the example row is unchanged.
   - the shapes that were already there keep their sizes, colours, and positions

3. Background Preservation: Whether the canvas stays clean.
   - the canvas around the shapes stays clean and unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
foreground_preservation_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
foreground_preservation_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-15_ball_bounces_given_time_data-generator': """
You are evaluating a generated video for a bouncing ball task.

Task: Animate a moderate bounce sequence. The ball should start from the initial position following the velocity arrow direction, demonstrate 4-5 bounces, creating an interesting path across the bounded area. Stop after the 5th bounce.

You will receive: the first frame (input) and the generated video.

Score the following 4 aspects (each 0-100):

1. Physics: Whether the ball's motion obeys the physics of bouncing.
   - it travels in straight lines between bounces
   - at each wall the angle it leaves equals the angle it arrived at

2. Trajectory: Whether the path the ball actually traces is the correct one.
   - the path follows where the ball should have gone, over the whole clip
   - it does not wander far off, nor cover much more ground than it should

3. Single Ball: Whether exactly one ball exists throughout.
   - the ball never duplicates or splits into several

4. Scene Fidelity: Whether the walls and the rest of the scene stay as they were.
   - the enclosure and any static markings are unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
physics_score: <0-100>
trajectory_score: <0-100>
ball_penalty_score: <0-100>
fg_similarity_score: <0-100>
physics_weight: <0-100>
trajectory_weight: <0-100>
ball_penalty_weight: <0-100>
fg_similarity_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-16_color_addition_data-generator': """
You are evaluating a generated video for an additive colour mixing task.

Task: Two colored circular balls start at different positions. They move toward each other at equal speeds until they fully overlap and merge into one. The overlapping region during movement and the final merged ball should show the additive color mixture of the two original ball colors. Stop the animation when the balls have completely merged at the midpoint.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Mixing Colour: Whether the two balls merge into one ball of the correct additive-mix colour.
   - the merged ball's colour is the right mix of the two originals
   - its colour is uniform rather than blotchy
   - its size and position match where and how big the merged ball should be

2. Originals Removed: Whether the two original balls are gone once they merge.
   - nothing is left behind where the two balls started

3. Background Clean: Whether the area outside the balls stays clean.
   - no stray colour, trails, or artifacts appear in the background

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
mixing_color_score: <0-100>
circle_removal_score: <0-100>
background_clean_score: <0-100>
mixing_color_weight: <0-100>
circle_removal_weight: <0-100>
background_clean_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-18_glass_refraction_data-generator': """
You are evaluating a generated video for a Snell's-law refraction task.

Task: Given the glass refractive index = 1.43, predict how light refracts when passing through the glass. Extend the refracted ray to the image edge.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Refracted Ray: Whether the drawn ray is the correct refracted ray.
   - it starts at the point where the incident ray meets the glass surface
   - it bends by the angle the refractive index dictates, in the right direction
   - it runs into the glass as one continuous straight segment of sensible length

2. Consistency: Whether the rest of the scene is unchanged.
   - the incident ray, the normal line, and the glass boundary stay exactly as they were
   - the background around them stays clean

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
red_line_score: <0-100>
consistency_score: <0-100>
red_line_weight: <0-100>
consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-19_mirror_reflection_data-generator': """
You are evaluating a generated video for a mirror reflection task.

Task: Given the mirror reflectivity = 0.43, predict how light reflects when it encounters the mirror. Extend the reflected ray to the image boundary.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Reflected Ray: Whether the drawn ray is the correct reflected ray.
   - it starts at the point where the incident ray meets the mirror
   - the angle it leaves at equals the angle the incident ray arrived at, measured from the normal
   - it runs away from the mirror as one continuous straight segment of sensible length

2. Consistency: Whether the rest of the scene is unchanged.
   - the incident ray, the normal line, and the mirror stay exactly as they were
   - the background around them stays clean

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
red_line_score: <0-100>
consistency_score: <0-100>
red_line_weight: <0-100>
consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-21_construction_blueprint_data-generator': """
You are evaluating a generated video for a blueprint missing piece task.

Task: Identify which of the four candidate pieces correctly fills the highlighted gap in the structure. The correct piece should match the gap's exact shape. Animate the selection process and final placement.

You will receive: the first frame (input) and the generated video.

Score the following 4 aspects (each 0-100):

1. Gap Filled Correctly: Whether the gap in the block structure ends up filled by the piece that fits it.
   - the red-dotted gap is completely filled
   - the piece placed there has the shape the gap calls for
   - no extra blocks are added outside the structure

2. Correct Option Marked: Whether the right candidate among the options is marked as chosen.
   - the candidate that fits the gap is marked in green

3. Other Options Marked: Whether the candidates that do not fit are marked as rejected.
   - the wrong candidates carry the rejected styling rather than being left unmarked or marked as chosen

4. No Unexpected Changes: Whether anything in the scene changed that the task did not ask for.
   - no option tile is erased or corrupted
   - the existing block structure is not disturbed

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
shape_matching_score: <0-100>
correct_option_green_score: <0-100>
row_score_score: <0-100>
deduction_score: <0-100>
shape_matching_weight: <0-100>
correct_option_green_weight: <0-100>
row_score_weight: <0-100>
deduction_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-23_domino_chain_branch_path_prediction_data-generator': """
You are evaluating a generated video for a Y-shaped domino chain task.

Task: Push the green START domino to begin the chain reaction. Show the complete domino falling process. The dominos fall step by step from left to right. The light gray trunk falls first, then splits into the pink Branch A (upward) and indigo Branch B (downward). Both branches complete successfully The final state shows all dominos fallen and tilted right.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Final State: Whether the right dominoes are lying down and the right ones are still standing at the end.
   - each domino that should have toppled is fallen, and each that should not is upright
   - in particular, a branch cut off by an oversized gap keeps its dominoes standing
   - every domino stays where it was placed; the background is unchanged

2. Process: Whether the toppling happens as a chain reaction.
   - the dominoes fall in sequence, propagating outward from the push, not all at once
   - the chain stops at any gap too wide to bridge, rather than jumping across it
   - the toppling runs to completion

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_state_score: <0-100>
process_score: <0-100>
final_state_weight: <0-100>
process_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-24_domino_chain_gap_analysis_data-generator': """
You are evaluating a generated video for a domino chain with a gap task.

Task: Analyze the domino chain to find which domino is the last to fall. Push the first domino and watch as each domino falls in sequence. The chain stops at the gap where the spacing between two upright dominos is visibly wider than the others.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Final State: Whether the chain has fallen exactly up to the gap and no further.
   - every domino before the wide gap is fallen
   - every domino after it is still standing
   - every domino stays where it was placed; the background is unchanged

2. Process: Whether the toppling happens as a left-to-right chain reaction.
   - the dominoes fall one after another in order, not simultaneously
   - the chain stops at the wide gap rather than jumping across it

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_state_score: <0-100>
process_score: <0-100>
final_state_weight: <0-100>
process_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-25_LEGO_construction_assembly_data-generator': """
You are evaluating a generated video for a LEGO assembly task.

Task: Add this brick to continue the wall construction. Follow the arrow to see where it connects to the existing structure.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Assembly Correctness: Whether the loose brick ends up correctly seated at the place the instructions indicate.
   - the brick sits at the target position, aligned to the studs beneath it
   - it has the right orientation rather than being rotated or flipped
   - the instruction diagram has been consumed, as it is in the intended result

2. Consistency: Whether the rest of the scene is unchanged.
   - the partial model that was already built keeps its bricks in place
   - the background outside the model stays clean

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
assembly_correctness_score: <0-100>
consistency_score: <0-100>
assembly_correctness_weight: <0-100>
consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-29_ballcolor_data-generator': """
You are evaluating a generated video for a cluster merging task.

Task: Show the red cluster (labeled A) moving toward other colored clusters (B, C, D). Only the red cluster moves; all other clusters remain stationary. When the red cluster collides with a target cluster, compare their counts. If the red cluster's count is greater than or equal to the target's count, the target disappears and the red cluster merges with it, keeping its red color and ID A. The red cluster's count label updates to show the exact sum. Continue until exactly one red cluster A remains, containing all balls.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Final State: Whether the scene ends with a single surviving cluster in the survivor's colour.
   - all balls carry the survivor cluster's colour
   - the total number of balls is what it should be — none created, none lost

2. Merge Process: Whether each absorption happens as it should.
   - the survivor cluster travels to a cluster and absorbs it, one at a time
   - at each merge the absorbed cluster's balls take on the survivor's colour
   - the ball counts before and after each merge are consistent

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_state_score: <0-100>
merge_process_score: <0-100>
final_state_weight: <0-100>
merge_process_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-30_bookshelf_data-generator': """
You are evaluating a generated video for a bookshelf insertion task.

Task: You have a bookshelf with gold books (already placed) and azure books (to be inserted). Each gold book has a height, and they are arranged from left to right. gold books are clustered by height: if two adjacent gold books have similar heights, they belong to the same cluster. For each azure book, find the gold book cluster whose representative height (average of all books in that cluster) is closest to the azure book's height. Insert each azure book at the end of its assigned cluster. If multiple azure books are assigned to the same position, insert them in order of increasing height. Output the 0-based insertion position for each azure book.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Final Placement: Whether every new book ends up in its correct slot on the shelf.
   - each inserted book sits in the gap it belongs in
   - the staging area the books came from is left empty

2. Sequential Insertion: Whether the books are inserted gradually, one at a time.
   - the shelf fills up progressively across the video rather than all books appearing at once

3. Consistency: Whether the rest of the scene is unchanged.
   - the books that were already on the shelf stay where they were
   - the area outside the books stays clean

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_placement_score: <0-100>
sequential_insertion_score: <0-100>
consistency_score: <0-100>
final_placement_weight: <0-100>
sequential_insertion_weight: <0-100>
consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-31_ball_eating_data-generator': """
You are evaluating a generated video for a greedy ball eating task.

Task: Animate the black ball using a greedy largest-first strategy to eat all red balls. At each step, the black ball targets the largest red ball that is smaller than or equal to its current size. After eating each red ball, the black ball grows larger (1.4x). Show smooth movement as the black ball approaches each target, the red ball disappearing when eaten, and the black ball growing after each consumption. Continue until all red balls are eaten.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Final State: Whether all the target balls are gone at the end and the black ball has grown accordingly.
   - no target ball remains
   - the black ball's final size reflects everything it consumed

2. Eating Process: Whether each ball is eaten in the right way and the right order.
   - the black ball eats the balls one at a time, smallest first among those it is big enough to eat
   - it moves to each ball and overlaps it at the moment that ball disappears
   - it grows after each meal

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_state_score: <0-100>
eat_process_score: <0-100>
final_state_weight: <0-100>
eat_process_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-32_rolling_ball_data-generator': """
You are evaluating a generated video for a rolling ball along a path task.

Task: Animate the ball rolling along its continuous trajectory path, smoothly transitioning from one platform to the next. The ball should follow the curved path through 3D space, ending at rest on the final platform.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Completion: Whether the ball rolls along the marked path and ends at the right place.
   - it stays on the dashed path rather than cutting across the scene
   - it keeps moving forward along the path rather than reversing
   - it comes to rest at the path's end point, at the right size

2. Consistency: Whether the rest of the scene is unchanged.
   - the dashed path and the rest of the scene are not erased or distorted
   - no extra objects appear

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
consistency_score: <0-100>
completion_weight: <0-100>
consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-33_counting_object_data-generator': """
You are evaluating a generated video for an object counting task.

Task: The scene shows circular objects scattered across the image. Each object is a filled circle with a black outline. Starting from any position in the image, systematically count all the circular objects visible in the scene. Count each object exactly once, ensuring that no object is missed or counted multiple times. After completing the count, display the total number of circular objects found in the scene.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Count Correctness: Whether the number written into the frame equals the true number of shapes.
   - a clearly legible number is drawn
   - that number is the correct count of the distinct shapes in the scene

2. Consistency: Whether the scene is unchanged apart from the written number.
   - the shapes stay where they are, unaltered
   - the background outside the shapes and the number stays clean

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
count_correctness_score: <0-100>
consistency_score: <0-100>
count_correctness_weight: <0-100>
consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-34_dot_to_dot_task_data-generator': """
You are evaluating a generated video for a dot-to-dot task.

Task: The scene shows 5 numbered dots scattered across the image. Connect the dots in numerical order (1→2→3→...→5) by drawing red straight lines between them, one line at a time in sequence.

You will receive: the first frame (input) and the generated video.

Score the following 4 aspects (each 0-100):

1. Connection Completeness: Whether every consecutive pair of numbered dots gets joined by a red segment.
   - each of the required segments is drawn
   - no spurious extra segments are drawn

2. Connection Order: Whether the segments are drawn in ascending numerical order.
   - the drawing proceeds 1 to 2, then 2 to 3, and so on
   - segments drawn out of order, or between non-consecutive dots, count against this

3. Numbers Intact: Whether the printed numbers beside the dots stay legible and unchanged.
   - no digit changes, disappears, or drifts during the video

4. Scene Consistency: Whether everything other than the dots and the new red lines is unchanged.
   - the background stays clean, with no smearing or stray marks

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
connection_completeness_score: <0-100>
connection_order_penalty_score: <0-100>
numerical_consistency_penalty_score: <0-100>
mask_region_similarity_score: <0-100>
connection_completeness_weight: <0-100>
connection_order_penalty_weight: <0-100>
numerical_consistency_penalty_weight: <0-100>
mask_region_similarity_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-36_grid_shift_data-generator': """
You are evaluating a generated video for a grid shift task.

Task: The scene displays a 6x6 grid containing 3 blue square blocks with black borders, distributed across different cells. Move every block rightward by precisely 1 step. All blocks move together at the same time, shifting 1 grid cell in the right direction, and each block must stay within the grid's boundaries. The final configuration shows all blocks in their new positions, each exactly 1 step rightward from where it started.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Final Cells: Whether the blocks end up in the cells the shift demands.
   - every occupied cell has moved by the same integer offset, in the direction and by the number of steps asked for
   - the number of occupied cells at the end equals the number at the start

2. Process: Whether the blocks get there by shifting gradually.
   - the blocks move step by step across the video rather than snapping to the final layout
   - the block count stays constant throughout

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_cell_score_score: <0-100>
process_score_score: <0-100>
final_cell_score_weight: <0-100>
process_score_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-37_light_sequence_data-generator': """
You are evaluating a generated video for a light state control task.

Task: The scene shows 10 circular lights in a horizontal row on a white background. Lights on are gold/yellow with glow; lights off are gray. Initially, some lights are on and some are off. Your task: Modify the light states so that the 1st, 4th, 7th, and 9th lights from the left (counting from left to right) are on (gold/yellow with glow), and all other lights are off (gray). Turn lights on/off as needed. Lights change from gray to gold/yellow (with glow) when turned on, and from gold/yellow to gray (glow disappears) when turned off. Lights stay in fixed positions; only their states change.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Completion: Whether exactly the right lights end up switched on.
   - the lights that should be on show the lit colour
   - the lights that should stay off remain grey

2. Background Preservation: Whether the scene around the lights is unchanged.
   - the lights stay in their places at their sizes, and the background stays clean

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-38_majority_color_data-generator': """
You are evaluating a generated video for a majority colour task.

Task: Identify the color that appears most frequently. Keep objects of the majority color unchanged and visible, while fading out all objects of other colors until they disappear.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Completion: Whether only the majority-colour shapes remain.
   - the shapes in the most common starting colour are still there, unchanged
   - every shape of any other colour has been removed, leaving no remnant

2. Background Preservation: Whether the background stays clean.
   - nothing is left behind where the removed shapes used to be

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-44_rotation_puzzle_data-generator': """
You are evaluating a generated video for a pipe rotation puzzle task.

Task: Solve this rotation puzzle by rotating the four squares to connect the pipe paths. Each square can be rotated 90 degrees clockwise or counterclockwise. Rotate the squares so that all pipe paths connect to form a continuous path. Keep the camera view fixed in the top-down perspective and maintain all square positions unchanged. Stop the video when all pipes are connected and the puzzle is solved.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Tile Integrity: Whether each tile stays the same L-shaped pipe throughout, only turning.
   - through every frame each cell holds the same pipe piece, merely rotated
   - no tile is deformed, swapped with another, or moved out of its cell

2. Completion: Whether the final tile orientations connect the pipes into one continuous path.
   - each tile ends at the rotation the solution requires
   - the pipe openings line up across tile boundaries

3. Background Preservation: Whether the grid and background are unchanged.
   - the grid lines and the area around the tiles stay as they were

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
template_preservation_score: <0-100>
completion_score: <0-100>
background_preservation_score: <0-100>
template_preservation_weight: <0-100>
completion_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-45_sequence_completion_data-generator': """
You are evaluating a generated video for a sequence completion task.

Task: The scene shows a color_cycle sequence. Elements are arranged horizontally from left to right. The last position contains a question mark (?) indicating a missing element. Observe the pattern: the colors follow a cyclic order that repeats after a certain number of elements. Determine the element that should replace the question mark to complete the sequence according to the established pattern.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Generated Element: Whether the element drawn into the unknown slot is the one the pattern demands.
   - it replaces the question mark
   - its colour, shape, and size are what the pattern implies

2. Consistency: Whether the rest of the sequence is unchanged.
   - the elements that were already given keep their positions and appearance
   - the background stays clean

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
generated_object_score: <0-100>
consistency_score: <0-100>
generated_object_weight: <0-100>
consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-47_sliding_puzzle_data-generator': """
You are evaluating a generated video for a 3x3 sliding puzzle task.

Task: Complete this sliding puzzle. The goal is to arrange the numbered tiles in grid order (filling each row from left to right, with rows from top to bottom), with the empty space at the bottom-right corner. Rules: Only tiles adjacent to the empty space can be moved. Slide one tile per move into the empty space. Complete in exactly 5 moves. Do not make extra moves. Keep the camera view fixed and maintain the grid structure unchanged.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Final State: Whether the puzzle ends solved.
   - the tiles read 1 through 8 in order, with the blank in the bottom-right cell

2. Process: Whether the puzzle is solved by legal sliding moves.
   - each move slides one tile into the adjacent blank cell
   - no tile jumps diagonally, swaps, or teleports
   - the number of moves is reasonable rather than endless thrashing

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_state_score: <0-100>
process_score: <0-100>
final_state_weight: <0-100>
process_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-52_traffic_light_data-generator': """
You are evaluating a generated video for a traffic light countdown task.

Task: This scene shows a crossroad with four traffic lights (North, South, East, West). Each light independently follows a 3-color cycle: Red (4s) → Yellow (4s) → Green (4s) → Yellow (4s) → Red. Currently: North light is red with 4s countdown, South light is green with 3s countdown, East light is red with 3s countdown, West light is red with 2s countdown. Simulate 5 seconds and show the final state of all four traffic lights.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Final State: Whether the lights end in the state that running the countdown produces.
   - each light shows the correct colour at the end
   - each countdown shows the correct final number
   - the rest of the crossroad scene is unchanged

2. Process: Whether the countdown runs and the lights switch in step with it.
   - the numbers decrease at a steady rate rather than skipping or freezing
   - each light changes colour exactly when its countdown reaches zero
   - the scene stays stable across the whole video

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_state_score: <0-100>
process_score: <0-100>
final_state_weight: <0-100>
process_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-53_clock_data-generator': """
You are evaluating a generated video for a clock hand rotation task.

Task: The clock shows 10:56. Show what the clock will look like after 21 hours.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Completion: Whether the hands end at the time that advancing by the given number of hours produces.
   - the hour hand points at the right hour, and the minute hand at the right minute
   - there is exactly one hour hand and one minute hand at the end

2. Process Validity: Whether the hands get there by turning clockwise.
   - both hands rotate clockwise throughout, never backwards
   - the hour hand sweeps through the full rotation the elapsed hours require
   - each hand stays a single, continuously visible hand

3. Element Preservation: Whether the clock face is unchanged.
   - the dial, numbers, and tick marks stay as they were
   - the area outside the clock stays clean

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
process_validity_score: <0-100>
element_preservation_score: <0-100>
completion_weight: <0-100>
process_validity_weight: <0-100>
element_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-55_rotation_data-generator': """
You are evaluating a generated video for a 3D mental rotation task.

Task: A 8-block sculpture sits fixed on a table. First frame: Your camera is tilted at 23° elevation, viewing from 338° azimuth. Final frame: Your camera remains at 23° elevation, but rotates horizontally to 158° azimuth. This is a 180-degree rotation Create a smooth video showing the camera's horizontal rotation around the sculpture, and try to maintain the tilted viewing angle throughout.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Final State: Whether the last frame shows the voxel structure seen from the rotated viewpoint.
   - the structure is the same object, viewed from the other side
   - no voxels are added, dropped, or rearranged

2. Process: Whether the camera sweeps smoothly around to that viewpoint.
   - the intermediate frames show the structure at the intermediate viewing angles
   - the rotation is continuous rather than a jump cut

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_state_score: <0-100>
process_score: <0-100>
final_state_weight: <0-100>
process_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-75_communicating_vessels_data-generator': """
You are evaluating a generated video for a communicating vessels task.

Task: Predict the fluid motion in a 3-tube apparatus. Initial heights are 5, 23, 16 cm. Based on gravity 9.8 m/s² and the damping k=1.72, generate a video showing the liquid equalizing to a common level.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Final Equilibrium: Whether the liquid ends level across all the connected vessels.
   - the liquid surfaces all settle at the same height
   - that height is the one the starting volume implies

2. Volume Conservation: Whether the total amount of liquid stays constant.
   - liquid flows between vessels without any appearing or disappearing

3. Consistency: Whether the apparatus is unchanged.
   - the vessel walls, connecting tubes, and any markings stay as they were
   - no extra objects appear

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_liquid_score: <0-100>
volume_conservation_score: <0-100>
consistency_score: <0-100>
final_liquid_weight: <0-100>
volume_conservation_weight: <0-100>
consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-135_select_next_figure_small_large_alternating_sequence_data-generator': """
You are evaluating a generated video for a circle-the-next-figure task.

Task: A sequence of shapes arranged in a 'small-big-small' pattern. Circle the next shape in the candidate area that continues this 'small-big-small-big' pattern.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the existing shapes keep their type, size, and position
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle marks the candidate that continues the small-big-small-big alternation.
   - the circle encloses the correct candidate
   - no wrong candidate is circled — circling a wrong one subtracts from the score
   - each circle unambiguously picks out one shape rather than straddling several

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-193_draw_next_sized_shape_data-generator': """
You are evaluating a generated video for a draw the next shape in the size pattern task.

Task: A group of shapes are arranged in a 'large-medium-small-large-medium' pattern, draw the next shape in the designated area.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Completion: Whether the new shape drawn in the empty rightmost slot is the one the pattern demands.
   - its size continues the recurring size pattern
   - its shape type, colour, and position within the slot match what is expected

2. Foreground Preservation: Whether the shapes that were already there are unchanged.
   - the existing shapes keep their sizes, colours, and positions

3. Background Preservation: Whether the canvas around the shapes stays clean.
   - the area outside the shapes is unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
foreground_preservation_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
foreground_preservation_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-136_locate_point_in_overlapping_area_data-generator': """
You are evaluating a generated video for a mark points in the overlap task.

Task: Two geometric shapes (possibly circles, rectangles, triangles, or polygons) are randomly generated on the canvas and overlap partially. Some points are scattered in the image; circle those that fall within the overlapping region of the two shapes.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circles changes.
   - the white background stays clean
   - the two shapes and the dots keep their positions
   - the red circles are thin annotations, not large red blobs

2. Selection Match: Whether every point lying inside the overlap of the two shapes is circled, and only those.
   - each point inside the overlap region is enclosed by a red circle
   - points outside the overlap are left unmarked
   - the circles are sized to the points rather than sprawling over the scene

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-140_locate_topmost_unobscured_figure_data-generator': """
You are evaluating a generated video for a locate the topmost figure task.

Task: Multiple shapes partially overlap. Outline the topmost (unobscured) shape.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Marking Accuracy: Whether the red outline traces the shape that lies on top of the overlapping pile (the one nothing else covers).
   - the outline follows that shape's boundary
   - exactly one shape is outlined; missing and extra outlines both count against this

2. Background Consistency: Whether the white background stays untouched.
   - the white canvas outside the shapes stays clean

3. Foreground Consistency: Whether all the original overlapping shapes are unchanged.
   - every shape keeps its colour, position, size, and stacking order

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
accuracy_score: <0-100>
back_consistency_score: <0-100>
fore_consistency_score: <0-100>
accuracy_weight: <0-100>
back_consistency_weight: <0-100>
fore_consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-147_identify_unique_figure_in_uniform_set_data-generator': """
You are evaluating a generated video for a circle the odd one out task.

Task: A group of shapes with consistent size, circle the only one that is different.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the existing shapes keep their type, size, and position
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle marks the single shape that differs from the otherwise-identical set.
   - the circle encloses that shape
   - no other shape is circled — circling a wrong one subtracts from the score
   - the circle unambiguously picks out one shape rather than straddling several

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-160_circle_largest_numerical_value_data-generator': """
You are evaluating a generated video for a circle the largest number task.

Task: There are multiple numbers on the screen, circle the one with the largest value

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the printed numbers keep their glyphs and positions
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle marks the number with the largest value.
   - the circle encloses that number
   - no other number is circled — circling a wrong one subtracts from the score
   - the circle unambiguously picks out one number rather than straddling several

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-161_mark_second_largest_shape_data-generator': """
You are evaluating a generated video for an outline the second-largest shape task.

Task: Three circles are shown, increasing in size from left to right. Please mark the second largest circle with a red outline.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Consistency: Whether nothing but the new outline changes.
   - the white background stays clean
   - the existing shapes keep their type, size, and position
   - the annotation is a thin outline, not a large filled blob

2. Selection Match: Whether the annotation marks the shape with the second-largest area.
   - it encloses that shape
   - no other shape is marked — marking a wrong one subtracts from the score
   - the annotation unambiguously picks out one shape

3. Outline Fit: How tightly the drawn outline hugs the target shape.
   - the outline follows the target's boundary rather than loosely surrounding it

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
shape_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
shape_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-167_select_longest_polygon_side_data-generator': """
You are evaluating a generated video for a mark the longest polygon side task.

Task: Each side of the polygon has a different length. First compare the lengths of all polygon edges, then mark the single longest side by drawing a small circle at its midpoint. Do not change anything else. Show the complete solution step by step.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the new mark changes.
   - the white background stays clean
   - the polygon keeps its vertices, edges, and position
   - the mark is a small annotation, not a large blob

2. Selection Match: Whether the small circle is drawn at the midpoint of the polygon's longest edge.
   - it sits on that edge's midpoint
   - no other edge is marked — marking a wrong one subtracts from the score
   - the mark unambiguously picks out one edge rather than straddling two

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-174_arrange_circles_by_circumference_data-generator': """
You are evaluating a generated video for an arrange circles by size task.

Task: Display circles moving to align on the horizontal line, with the largest circumference first.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Arrangement: Whether the circles end up ordered by size along a line.
   - reading left to right, the circles run from largest to smallest
   - their centres sit on a common horizontal line

2. Background Consistency: Whether the area outside the circles stays plain white.
   - no leftover marks or drag artifacts remain where circles used to be

3. Circle Fidelity: Whether the set of circles is exactly the set that started.
   - every original circle is still present with its own size
   - no extra or duplicated circles have appeared

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
arrangement_score: <0-100>
back_consistency_score: <0-100>
fore_consistency_score: <0-100>
arrangement_weight: <0-100>
back_consistency_weight: <0-100>
fore_consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-202_mark_wave_peaks_data-generator': """
You are evaluating a generated video for a mark the wave peaks task.

Task: Animate a multi-peaked wave and identify all peak positions. Use red ring markers with filled center dots to mark each local maximum clearly.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circles changes.
   - the white background stays clean
   - the wave curve keeps its shape and position
   - the red circles are thin annotations, not large red blobs

2. Selection Match: Whether every peak of the wave is circled, and only the peaks.
   - each local maximum of the curve is enclosed by a red circle
   - troughs and mid-slope points are left unmarked
   - the circles are sized to the peaks rather than sprawling over the curve

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-212_find_incorrect_arrow_direction_data-generator': """
You are evaluating a generated video for a circle the odd arrow task.

Task: Observe this circular arrangement of arrows and circle the one that points in a different direction from the rest.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red mark changes.
   - the white background stays clean
   - the arrows keep their positions and directions
   - the mark is a small annotation, not a large blob

2. Selection Match: Whether the mark identifies the one arrow pointing in a different direction from the rest.
   - it lands on that arrow
   - no correctly-pointing arrow is marked — marking a wrong one subtracts from the score
   - the mark unambiguously picks out one arrow rather than straddling two

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-217_circle_central_dot_data-generator': """
You are evaluating a generated video for a circle the central dot task.

Task: On a white canvas, black dots form a vertical column. The red circular marker appears with a smooth fade to highlight the dot positioned at the canvas center. Circle that dot.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the column of black dots keeps its positions
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle marks the dot that sits at the canvas centre.
   - the circle encloses that dot
   - no other dot is circled — circling a wrong one subtracts from the score
   - the circle unambiguously picks out one dot rather than straddling several

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-218_identify_largest_angle_in_triangle_data-generator': """
You are evaluating a generated video for a circle the largest angle task.

Task: Mark the largest angle which is obtuse with a red circle.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the triangle keeps its vertices and edges
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle marks the triangle's obtuse (largest) angle.
   - the circle sits on that vertex
   - neither of the other two vertices is circled — circling a wrong one subtracts from the score
   - the circle unambiguously picks out one vertex

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-219_select_leftmost_shape_data-generator': """
You are evaluating a generated video for a circle the leftmost shape task.

Task: There are multiple shapes with the same form but different colors. Circle the leftmost one.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the shapes keep their colours, sizes, and positions
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle marks the shape furthest to the left.
   - the circle encloses that shape
   - no other shape is circled — circling a wrong one subtracts from the score
   - the circle unambiguously picks out one shape rather than straddling several

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-221_outline_innermost_square_data-generator': """
You are evaluating a generated video for an outline the innermost square task.

Task: The scene contains multiple concentric squares with random colors and sizes. Mark the innermost square using a blue square outline; the initial frame has no outline, and the marker appears with a smooth fade-in/transition.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Outlining Accuracy: Whether the blue outline traces the innermost square of the nested set.
   - the outline follows that square's boundary
   - exactly one square is outlined; missing and extra outlines both count against this

2. Foreground Consistency: Whether the nested squares themselves are unchanged.
   - every square keeps its size, position, and colour; only the blue outline is added

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
accuracy_score: <0-100>
fore_consistency_score: <0-100>
accuracy_weight: <0-100>
fore_consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-240_add_borders_to_unbordered_shapes_data-generator': """
You are evaluating a generated video for an add borders to unbordered shapes task.

Task: Multiple shapes (some with borders, some without). Add black thin borders to all shapes without borders.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Bordering Accuracy: Whether a black border is added around exactly the shapes that started without one.
   - each previously unbordered shape now has a border tracing its boundary
   - no border is drawn where there is no shape; missing and spurious new borders both count against this

2. Background Consistency: Whether the white background stays untouched.
   - the white canvas outside the shapes stays clean

3. Foreground Consistency: Whether the shapes' interiors are unchanged.
   - every shape keeps its fill colour, size, and position; only borders are added

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
accuracy_score: <0-100>
back_consistency_score: <0-100>
fore_consistency_score: <0-100>
accuracy_weight: <0-100>
back_consistency_weight: <0-100>
fore_consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-247_identify_chinese_character_data-generator': """
You are evaluating a generated video for a circle the Chinese character task.

Task: Find and circle the Chinese character among the displayed characters. Only one character is Chinese. Draw a red circle around it.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the printed characters keep their glyphs and positions
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle marks the one Chinese character among the others.
   - the circle encloses that character
   - no non-Chinese character is circled — circling a wrong one subtracts from the score
   - the circle unambiguously picks out one character rather than straddling several

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-248_mark_asymmetrical_shape_data-generator': """
You are evaluating a generated video for a circle the asymmetrical shape task.

Task: Identify and circle the asymmetrical shape among the displayed shapes. Only one shape lacks symmetry.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the shapes keep their forms, sizes, and positions
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle marks the single shape that lacks symmetry.
   - the circle encloses that shape
   - no symmetrical shape is circled — circling a wrong one subtracts from the score
   - the circle unambiguously picks out one shape rather than straddling several

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-273_high_density_liquid_data-generator': """
You are evaluating a generated video for a liquid density float/sink task.

Task: There are 4 cups of liquid; above each cup is one identical object (same mass and density), which will fall into the corresponding cup. One cup has a different liquid color from the others, and that cup's liquid color is lighter; that cup's liquid density is lower than the object, and the others are higher. By buoyancy: when the liquid density is higher than the object, the object floats; when lower, it sinks. Show where each object should be in each cup and the final floating or sunk state.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Final State: Whether each object ends at the height its liquid's density dictates.
   - an object in a liquid at least as dense as itself floats; a denser object sinks
   - each object comes to rest at the correct height in its own cup
   - the cups, liquids, and background are otherwise unchanged

2. Process: Whether the objects get there by falling plausibly.
   - each object passes through the intermediate heights rather than teleporting to its resting place
   - at most one object occupies each cup at any moment
   - each object keeps its size throughout
   - the intermediate frames stay clean and uncorrupted

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_state_score: <0-100>
process_score: <0-100>
final_state_weight: <0-100>
process_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-47_multiple_keys_for_one_door_data-generator': """
You are evaluating a generated video for a multi-key maze task.

Task: The scene shows a maze with a green circular agent, two colored diamond-shaped keys, and a red hollow rectangular door. Collect all keys along the shortest path (visiting all keys in the optimal order), then navigate to the red door along the shortest path, showing the complete movement process step by step.

You will receive: the first frame (input) and the generated video.

Score the following 7 aspects (each 0-100):

1. Path Correctness: Whether the green agent stays on corridor cells of an efficient route that collects every key and then reaches the door.
   - at every frame it sits on a cell of such a route

2. Path Completion: Whether the agent walks the whole route rather than stopping partway.
   - it progresses from the start, through the keys, to the red door

3. Path Efficiency: Whether the agent takes a direct route rather than zig-zagging.
   - the total distance travelled is close to the shortest possible tour of the keys and door

4. Motion Continuity: Whether the agent moves as one continuous object.
   - it advances cell by cell, without teleporting, vanishing, or flickering

5. All Keys Collected: Whether the agent visits every coloured diamond key.
   - each key never visited is a serious violation

6. Wall Avoidance: Whether the agent ever steps into a black wall cell.
   - every distinct wall cell it intrudes into is a serious violation

7. Background Preservation: Whether the maze itself stays as in the input frame.
   - walls, keys, and the door are unchanged wherever the agent is not standing

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
proximity_score: <0-100>
coverage_score: <0-100>
length_factor_score: <0-100>
continuity_factor_score: <0-100>
key_multiplier_score: <0-100>
wall_multiplier_score: <0-100>
bg_preservation_score: <0-100>
proximity_weight: <0-100>
coverage_weight: <0-100>
length_factor_weight: <0-100>
continuity_factor_weight: <0-100>
key_multiplier_weight: <0-100>
wall_multiplier_weight: <0-100>
bg_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-2_pigment_color_mixing_subtractive_data-generator': """
You are evaluating a generated video for a subtractive pigment mixing task.

Task: Two pigment colors are displayed. A rectangular zone marked with a white border indicates where the pigments will mix. Using subtractive color mixing rules, determine the resulting color and display it within the marked zone.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Mixing Colour: Whether the mixing box ends up showing the correct subtractive mix of the two pigment colours.
   - the colour in the box is the right mixed colour
   - the box is filled completely rather than partly
   - the fill is uniform rather than streaky or patchy

2. Object Preservation: Whether the two coloured circles are unchanged.
   - they keep their colours, sizes, and positions

3. Background Clean: Whether the area outside the circles and the box stays clean.
   - no stray colour or artifacts appear in the background

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
mixing_color_score: <0-100>
object_preservation_score: <0-100>
background_clean_score: <0-100>
mixing_color_weight: <0-100>
object_preservation_weight: <0-100>
background_clean_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-5_symbol_deletion_data-generator': """
You are evaluating a generated video for a symbol deletion task.

Task: The scene shows a horizontal sequence of colored geometric symbols arranged from left to right. Each symbol is a distinct geometric shape with a specific color, and exactly one symbol is marked as the deletion target by a red rectangular border surrounding it. In symbol deletion tasks, the target symbol identified by the red border must be deleted from the sequence while all remaining symbols maintain their original sequential order. First identify the symbol marked with the red border, then delete it from the sequence. The final state must show the remaining symbols in their original order, with the target symbol completely removed. The deletion operation affects only the single marked symbol, leaving all other symbols unchanged in their shapes, colors, and sequential positions.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Deletion Accuracy: Whether the symbol inside the red box is erased cleanly.
   - the region inside the red box ends up blank
   - no ghost, smear, or partial remnant of the deleted symbol is left

2. Kept Symbols: Whether every symbol outside the red box is left exactly as it was.
   - the other symbols keep their shapes and positions; none is moved, shifted, or altered

3. Background Consistency: Whether the area above and below the symbol row is unchanged.
   - the blank margins stay blank

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
delete_score: <0-100>
keep_score: <0-100>
bg_score: <0-100>
delete_weight: <0-100>
keep_weight: <0-100>
bg_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-6_2d_geometric_transformation_data-generator': """
You are evaluating a generated video for a rotation about a pivot task.

Task: Perform a 2D geometric transformation by rotating the planar object to the target position. You are given: (1) a solid-colored 2D shape in its starting orientation, (2) a clearly marked rotation center point, and (3) a target outline indicating where the shape should end up after rotation. Your task is to rotate the shape in the 2D plane around the specified rotation center until the shape's position precisely aligns with and overlaps the target outline. The object must maintain its size and shape during rotation, only its orientation should change.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Final Pose: Whether the shape ends up in the target pose.
   - its final position and orientation match the outline it is meant to land in
   - it keeps its size and shape rather than stretching or shrinking

2. Orbital Motion: Whether the shape gets there by genuinely rotating around the marked pivot.
   - it sweeps along a circular arc centred on the pivot, at a constant radius
   - it does not teleport straight to the target or drift on the wrong radius

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_pose_score: <0-100>
on_orbit_score: <0-100>
final_pose_weight: <0-100>
on_orbit_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-9_shape_scaling_data-generator': """
You are evaluating a generated video for a visual analogy task.

Task: Animate the scaling transformation where the shape changes size according to the established pattern.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Completion: Whether the bottom-right panel is filled with the shape the analogy demands: the same shape type as the bottom-left one, resized by the same factor that relates the top two panels.
   - the drawn shape's size follows the top row's scaling relation
   - its shape type, colour, and position within the panel are right

2. Foreground Preservation: Whether the panels that were already filled in are unchanged.
   - the shapes that were already there keep their sizes, colours, and positions

3. Background Preservation: Whether the canvas stays clean.
   - the canvas around the shapes stays clean and unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
foreground_preservation_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
foreground_preservation_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-11_shape_color_then_move_data-generator': """
You are evaluating a generated video for a two-step colour-then-move task.

Task: Complete the analogy by revealing the shape with the correct color and position.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Completion: Whether both steps of the transformation are applied to the target shape.
   - the recoloured shape carries the colour the example demands
   - the moved shape ends at the position the example demands
   - each shape keeps its type and size

2. Foreground Preservation: Whether the example row is unchanged.
   - the shapes that were already there keep their sizes, colours, and positions

3. Background Preservation: Whether the canvas stays clean.
   - the canvas around the shapes stays clean and unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
foreground_preservation_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
foreground_preservation_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-22_construction_stack_data-generator': """
You are evaluating a generated video for a copy the block stack task.

Task: Rearrange the block stacks to match the target state. Only the topmost block can be moved at any time. Use the available stacks strategically to minimize moves. Plan your moves carefully before starting.

You will receive: the first frame (input) and the generated video.

Score the following 4 aspects (each 0-100):

1. Final Stack Match: Whether the left-hand stack ends up matching the target stack on the right.
   - the same number of stacks, each with the same blocks in the same top-to-bottom colour order
   - read left to right, the stacks line up with the target's

2. Valid Movements: Whether the blocks are actually moved, one lift-and-place at a time.
   - the video shows blocks being picked up and set down rather than snapping into place
   - a video that never shows any block moving is worthless, however the final frame looks
   - each move takes a block from the top of a stack and places it on a grounded stack

3. Target Untouched: Whether the right-hand target stack stays put for the whole video.
   - it keeps its blocks, colours, and order from the first frame to the last

4. Background Clean: Whether the area outside the stacks stays clean.
   - no stray blocks, trails, or artifacts appear

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
main_score_score: <0-100>
movement_validity_ratio_score: <0-100>
deduction_target_frames_score: <0-100>
deduction_background_score: <0-100>
main_score_weight: <0-100>
movement_validity_ratio_weight: <0-100>
deduction_target_frames_weight: <0-100>
deduction_background_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-39_maze_data-generator': """
You are evaluating a generated video for a maze pathfinding task.

Task: The scene contains a 15x15 cell maze of medium complexity, featuring black walls and white pathways. A green circular marker shows the start position, and a red flag shows the end position. From the green start, navigate through the maze by traversing adjacent white cells in four directions (up, down, left, right), keeping clear of black wall cells. Discover and animate the full route from the green start to the red flag end, displaying each cell visited along the solution path.

You will receive: the first frame (input) and the generated video.

Score the following 4 aspects (each 0-100):

1. Path Correctness: Whether the traced route follows a valid path through the maze from the green start to the red end.
   - the route stays on the cells of a correct solution path rather than wandering

2. Path Completion: Whether the whole route is traced, start to end.
   - the path runs from the green start marker all the way to the red end flag

3. Motion Continuity: Whether the path is drawn as one continuous progression.
   - it advances cell by cell, without teleporting, vanishing, or flickering

4. Wall Avoidance: Whether the route ever crosses a black wall.
   - every place where the path cuts through a wall is a serious violation

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
proximity_score: <0-100>
coverage_score: <0-100>
continuity_factor_score: <0-100>
wall_multiplier_score: <0-100>
proximity_weight: <0-100>
coverage_weight: <0-100>
continuity_factor_weight: <0-100>
wall_multiplier_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-43_object_subtraction_data-generator': """
You are evaluating a generated video for a selective deletion task.

Task: The scene shows multiple colored geometric objects (cubes, spheres, pyramids, cones) positioned across the canvas. Most objects share similar visual characteristics (color, shape, or both), while one object appears visually distinct or different from the others. Identify the object that looks different from the rest based on its visual attributes. Remove this distinct object from the scene by making it disappear completely. All other objects must remain stationary and unchanged in their exact original positions, colors, and shapes. Do not modify, move, or alter any objects other than the one that appears different.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Completion: Whether exactly the indicated objects are removed and the rest are left alone.
   - every object that should go is fully erased, with no ghost or partial remnant
   - every object that should stay keeps its position, colour, size, and orientation

2. Background Preservation: Whether the background stays clean.
   - nothing is left behind where the removed objects used to be, and no new object appears

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-46_shape_sorter_data-generator': """
You are evaluating a generated video for a shape sorter task.

Task: Solve the flat shape sorter puzzle exactly as shown. Starting from the unsolved first frame, drag the colored cards across the board and place them into the matching outlines on the right. Match the yellow triangle, blue star, and finally the red diamond card. Keep the board orientation unchanged and end once all outlines are packed tightly.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Final Layout: Whether every shape ends up on its matching outline.
   - each coloured shape sits on the outline of its own form
   - the left-hand tray ends up empty

2. Process: Whether the shapes are transported one at a time, cleanly.
   - each shape travels to its outline rather than teleporting
   - the shapes move one at a time rather than all at once

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
final_layout_score: <0-100>
process_score_score: <0-100>
final_layout_weight: <0-100>
process_score_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-49_symmetry_completion_data-generator': """
You are evaluating a generated video for a symmetry completion task.

Task: Complete this pattern by filling in the missing grid cells on the right side. Observe the left half of the pattern and recognize that it should be mirrored to create a symmetric pattern. Fill in the right half by mirroring the left half across the vertical center line. Keep the camera view fixed in the top-down perspective and maintain all existing cells unchanged. Stop the video when the symmetric pattern is fully completed.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Completion: Whether the newly filled cells are exactly those that make the pattern symmetric.
   - each cell that must be filled to mirror an existing block is filled, in the mirrored block's colour
   - cells that should stay blank stay blank

2. Consistency: Whether the original blocks and the grid are unchanged.
   - every originally filled cell keeps its colour
   - the grid lines and outer background stay as they were

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
consistency_score: <0-100>
completion_weight: <0-100>
consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-54_control_panel_data-generator': """
You are evaluating a generated video for a control panel task.

Task: The image shows a control panel with three identical control units. Each unit has a colored indicator light at the top and a control lever at the bottom that can be moved to three positions (left, middle, or right). Observe the current control panel to infer the relationship between lever positions and light colors. Based on this inferred relationship, adjust the levers that need to be changed to make all indicator lights show pink color.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Completion: Whether every control ends in its specified target state.
   - each light shows its target colour
   - each lever sits at its target position

2. Process Validity: Whether the controls change state plausibly.
   - the levers slide and the lights change over the course of the video rather than snapping instantly
   - no control flickers between states

3. Background Preservation: Whether the panel surface around the controls is unchanged.
   - the empty areas of the panel stay exactly as they were, with no smearing or stray marks

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
process_validity_score: <0-100>
background_preservation_score: <0-100>
completion_weight: <0-100>
process_validity_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-56_raven_data-generator': """
You are evaluating a generated video for a Raven's matrix task.

Task: This is Raven's Progressive Matrices like task. Complete the missing pattern in this 3x3 matrix.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Completion: Whether the bottom-right cell is filled with the figure that completes the matrix.
   - the drawn figure matches what the row and column rules imply
   - the cell is not left empty

2. Preservation: Whether the other eight cells are unchanged.
   - each given cell keeps its figures exactly as they were

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
completion_score: <0-100>
preservation_score: <0-100>
completion_weight: <0-100>
preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-58_symbol_delete_data-generator': """
You are evaluating a generated video for a symbol delete and shift task.

Task: Delete orange ♥ at position 4. The animation shows the target symbol fading out, then the remaining symbols shifting left to close the gap.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Sequence Correctness: Whether the target symbol is deleted and the rest shift left to close the gap.
   - the symbol that should go is gone
   - the remaining symbols move left into the correct slots, leaving no hole
   - the outer boxes and the printed numbers stay exactly where they were

2. Background Consistency: Whether the area outside the symbol row is unchanged.
   - the blank margins stay blank

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
fg_score: <0-100>
bg_score: <0-100>
fg_weight: <0-100>
bg_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-59_symbol_insert_data-generator': """
You are evaluating a generated video for a symbol sequence edit task.

Task: Insert a violet ◯ at position 6. A reference panel in the top-right shows the target symbol. The animation shows the new symbol fading in above the target position, then sliding down while other symbols shift to make room.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Sequence Correctness: Whether the new symbol is inserted at the right place and the rest shift to make room.
   - the inserted symbol matches the reference template and lands in the correct slot
   - the symbols after it shift along correctly, leaving no gap or overlap
   - the outer boxes and the printed numbers stay exactly where they were

2. Template Intact: Whether the reference shape in the top-right corner is untouched.
   - it keeps its shape and position; it is neither distorted nor deleted

3. Background Consistency: Whether the area outside the sequence and the template is unchanged.
   - the blank margins stay blank

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
seq_score: <0-100>
template_score: <0-100>
bg_score: <0-100>
seq_weight: <0-100>
template_weight: <0-100>
bg_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-60_symbol_substitute_data-generator': """
You are evaluating a generated video for a symbol sequence edit task.

Task: Substitute ▼ at position 5 with a green ★. A reference panel in the top-right shows the target symbol. The animation shows the old symbol fading out completely, then the new symbol gradually fading in at the same position.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Sequence Correctness: Whether the target symbol is replaced by the new one and nothing else moves.
   - the symbol at the target slot is replaced by one matching the reference template
   - every other symbol keeps its slot and appearance
   - the outer boxes and the printed numbers stay exactly where they were

2. Template Intact: Whether the reference shape in the top-right corner is untouched.
   - it keeps its shape and position; it is neither distorted nor deleted

3. Background Consistency: Whether the area outside the sequence and the template is unchanged.
   - the blank margins stay blank

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
seq_score: <0-100>
template_score: <0-100>
bg_score: <0-100>
seq_weight: <0-100>
template_weight: <0-100>
bg_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-61_symbol_edit_data-generator': """
You are evaluating a generated video for a symbol sequence edit task.

Task: The sequence currently has 1 of symbol ★. Constraint: at least 4 of symbol ★. A reference panel in the top-right shows the target symbol. Insert 3 ★ symbols at positions 1, 2, and 6 to satisfy the constraint.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Sequence Correctness: Whether the sequence is edited exactly as the constraint requires.
   - the edit is applied at the correct slot, using a symbol matching the reference template
   - the other symbols shift or stay put as the edit demands, leaving no gap or overlap
   - the outer boxes and the printed numbers stay exactly where they were

2. Template Intact: Whether the reference shape in the top-right corner is untouched.
   - it keeps its shape and position; it is neither distorted nor deleted

3. Background Consistency: Whether the area outside the sequence and the template is unchanged.
   - the blank margins stay blank

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
seq_score: <0-100>
template_score: <0-100>
bg_score: <0-100>
seq_weight: <0-100>
template_weight: <0-100>
bg_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-62_gravity_physics_data-generator': """
You are evaluating a generated video for a gravity and bouncing task.

Task: A ball at height 22.7m with initial downward velocity 3.7 m/s falls under gravity 7.8 m/s² and bounces on ground with elasticity 0.80. Show the full trajectory with velocity arrows (direction and magnitude) updating throughout until the ball stops.

You will receive: the first frame (input) and the generated video.

Score the following 4 aspects (each 0-100):

1. Process: Whether the ball's fall and bounces unfold as gravity dictates.
   - the sequence of bounces and rebound peaks matches what should happen
   - each rebound reaches the height it should, with the bounces damping over time

2. Final Position: Whether the ball ends up where it should.
   - its resting place in the last frame is the physically correct one

3. Scene Preservation: Whether the static parts of the scene stay put.
   - the ground line and any other fixed elements are unchanged

4. No Lateral Drift: Whether the ball falls straight down.
   - it does not drift sideways as it falls and bounces

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
process_score: <0-100>
final_position_score: <0-100>
bg_penalty_score: <0-100>
lateral_penalty_score: <0-100>
process_weight: <0-100>
final_position_weight: <0-100>
bg_penalty_weight: <0-100>
lateral_penalty_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-64_animal_matching_data-generator': """
You are evaluating a generated video for an animal matching task.

Task: Move each animal face into its corresponding dark outline.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Placement: Whether each animal face ends up on its own matching silhouette.
   - each face lands on the silhouette of that same animal, not another one

2. Correct Count: Whether the right number of animals ends up placed.
   - as many animals arrive on the right as there were silhouettes to fill; none is duplicated or lost

3. Source Cleared: Whether the left-hand staging area is emptied.
   - no animal face is left behind on the left

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
placement_score: <0-100>
count_penalty_score: <0-100>
left_clear_score: <0-100>
placement_weight: <0-100>
count_penalty_weight: <0-100>
left_clear_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-65_animal_size_sorting_data-generator': """
You are evaluating a generated video for an animal size sorting task.

Task: Animal faces of different sizes are scattered randomly on the canvas. Sort them by size from smallest to largest and align them horizontally at the bottom baseline.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Arrangement: Whether the animals end up ordered by size along the baseline.
   - read left to right, the animals run in the intended size order
   - they all stand on the common baseline, with their centres level with one another

2. Animal Fidelity: Whether every animal that started is still findable at the end.
   - no animal has vanished or become unrecognisable

3. Background Consistency: Whether the area outside the animals and the baseline stays white.
   - no leftover marks or drag artifacts remain where animals used to be

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
arrangement_score: <0-100>
fore_consistency_score: <0-100>
back_consistency_score: <0-100>
arrangement_weight: <0-100>
fore_consistency_weight: <0-100>
back_consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-85_2d_object_rotation_data-generator': """
You are evaluating a generated video for a 2D object rotation task.

Task: The scene contains 2 2D object(s). Show them rotating counterclockwise by 43 degrees around their respective centroids.

You will receive: the first frame (input) and the generated video.

Score the following 3 aspects (each 0-100):

1. Rotation Trajectory: Whether each object turns through the intermediate angles rather than jumping.
   - at each moment the objects show the rotation reached so far
   - each object keeps its form and area while turning

2. Completion: Whether each object ends at the target orientation.
   - each object's final angle differs from its starting angle by the requested rotation, in the requested direction
   - each object's centre, colour, and size are unchanged

3. Background Preservation: Whether the background is unchanged.
   - the area around the objects stays clean

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
shape_preservation_score: <0-100>
completion_score: <0-100>
background_preservation_score: <0-100>
shape_preservation_weight: <0-100>
completion_weight: <0-100>
background_preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-24_separate_objects_no_spin_data-generator': """
You are evaluating a generated video for a separate objects without spinning task.

Task: The scene shows 2 objects on the left side and dashed target outlines on the right side. The dashed target outlines remain completely stationary. Move each object horizontally to the right so that it aligns exactly with and fits within its corresponding dashed target outline.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Alignment: Whether each filled shape ends up on its matching outline on the right.
   - each shape's final centre coincides with the centre of the outline it belongs to

2. Motion Quality: Whether the shapes get there by sliding cleanly, without side effects.
   - each shape slides horizontally rather than jumping
   - no teleporting between distant positions
   - each shape keeps its orientation — it must not rotate
   - no extra or hallucinated shapes appear in the final frame
   - the background stays unchanged

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
alignment_score: <0-100>
non_alignment_score_score: <0-100>
alignment_weight: <0-100>
non_alignment_score_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-54_connecting_color_data-generator': """
You are evaluating a generated video for a same-colour connection task.

Task: Animate smooth curves that connect each color's shapes without crossing other shapes or curves.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Correct Connections: Whether every pair of same-colour objects is joined by a curve of that same colour.
   - each same-colour pair is linked by a curve
   - each curve carries the colour of the objects it links
   - no curve joins objects of different colours

2. Consistency: Whether the scene is otherwise unchanged.
   - the original coloured objects stay in place, unchanged — if they are destroyed the whole score collapses
   - the white background outside the objects and the correct curves stays clean

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
correct_connections_score: <0-100>
consistency_score: <0-100>
correct_connections_weight: <0-100>
consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-168_identify_nearest_to_square_rectangle_data-generator': """
You are evaluating a generated video for a circle the most square-like rectangle task.

Task: Select the rectangle whose width-to-height ratio is nearest to 1:1. Draw a red circle around your chosen rectangle.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the rectangles keep their sizes and positions
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle marks the rectangle whose width-to-height ratio is closest to 1:1.
   - the circle encloses that rectangle
   - no other rectangle is circled — circling a wrong one subtracts from the score
   - the circle unambiguously picks out one rectangle rather than straddling several

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-169_locate_intersection_of_segments_data-generator': """
You are evaluating a generated video for a mark the segment intersection task.

Task: Animate two crossing line segments and circle their intersection with a red circle.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the line segments keep their endpoints and directions
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle is centred on the point where the segments cross.
   - it encloses the true crossing point
   - it does not mark some other point along the segments
   - the circle is sized to the point rather than sprawling over the scene

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-189_draw_midpoint_perpendicular_line_data-generator': """
You are evaluating a generated video for a draw the midpoint perpendicular task.

Task: Animate the process of drawing a perpendicular line through the middle point. The red line should grow smoothly from the middle point until it reaches both the upper and lower parallel lines.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Red Line Correctness: Whether the drawn red line is the perpendicular through the midpoint.
   - it lies where the perpendicular belongs: midway between the two reference points, at right angles to them
   - it spans the full gap between them rather than being a stub
   - no red ink appears anywhere outside that line

2. Consistency: Whether the rest of the scene is unchanged.
   - the dots and the black lines stay exactly where they were
   - the background outside them stays clean

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
red_line_score: <0-100>
consistency_score: <0-100>
red_line_weight: <0-100>
consistency_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-206_identify_pentagons_data-generator': """
You are evaluating a generated video for a circle the pentagon task.

Task: Multiple polygons are shown. Identify the only pentagon (5 sides) and mark it with a red circle that expands from the inside out.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the polygons keep their vertex counts, sizes, and positions
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle marks the five-sided polygon.
   - the circle encloses the pentagon
   - no polygon with a different number of sides is circled — doing so subtracts from the score
   - the circle unambiguously picks out one polygon rather than straddling several

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-222_mark_tangent_point_of_circles_data-generator': """
You are evaluating a generated video for a mark the tangent point task.

Task: Look at the circles on the screen. Please circle the tangent point of the two circles that are touching.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the red circle changes.
   - the white background stays clean
   - the two circles keep their centres and radii
   - the red circle is a thin annotation, not a large red blob

2. Selection Match: Whether the red circle is centred on the point where the two circles touch.
   - it encloses the true tangency point
   - it does not mark some other point on either circle
   - the circle is sized to the point rather than sprawling over the scene

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-223_highlight_horizontal_lines_data-generator': """
You are evaluating a generated video for a highlight the horizontal lines task.

Task: Observe these lines and circle all horizontal ones.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Consistency: Whether nothing but the new marks changes.
   - the white background stays clean
   - the lines keep their endpoints and orientations
   - the marks are small annotations, not large blobs

2. Selection Match: Whether exactly the horizontal lines are highlighted.
   - each horizontal line is marked
   - no slanted or vertical line is marked — marking a wrong one subtracts from the score
   - each mark unambiguously picks out one line rather than straddling two

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
consistency_score: <0-100>
match_score: <0-100>
consistency_weight: <0-100>
match_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'G-250_color_triple_intersection_red_data-generator': """
You are evaluating a generated video for a fill the triple intersection task.

Task: Look at the Venn diagram on the screen. Please color the area that belongs to the intersection of all three sets in red.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Red Region Correctness: Whether the red fill covers exactly the region where all three circles overlap.
   - the whole triple-overlap region is filled
   - the red does not spill into the pairwise overlaps or outside the circles

2. Preservation: Whether everything outside that region is unchanged.
   - the three circle outlines, the other overlap regions, and the background stay as they were

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
red_region_score: <0-100>
preservation_score: <0-100>
red_region_weight: <0-100>
preservation_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",

'O-27_move_2_object_to_2_target_data-generator': """
You are evaluating a generated video for a move objects onto targets task.

Task: Animate both balls sliding into their target areas. The movements should be smooth and direct, with each ball ending inside its corresponding dashed circle.

You will receive: the first frame (input) and the generated video.

Score the following 2 aspects (each 0-100):

1. Movement: Whether each ball travels to and lands on the target ring of its own colour.
   - each ball ends up on the ring matching its colour, not the other one
   - each ball actually travels there rather than staying put

2. Synchronisation: Whether the two balls move at roughly the same time.
   - they set off and arrive together rather than one moving long after the other

Then decide how much each aspect should count toward the overall quality of this
video, as percentages that sum to 100.

Output format (exactly these lines):
movement_score: <0-100>
synchronization_score: <0-100>
movement_weight: <0-100>
synchronization_weight: <0-100>
total_score: <0-100, your overall judgement of the video>
reason: <1-2 sentences>
""",
}

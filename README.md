# Exminationroom_dection

## Structure

```mermaid
%%{init: {'theme': 'default', 'themeVariables': {'background': '#ffffff', 'primaryColor': '#ffffff', 'primaryBorderColor': '#333333', 'lineColor': '#333333', 'fontSize': '16px', 'fontFamily': 'Times New Roman', 'primaryBorderWidth': '3px'}}}%%
flowchart LR
    classDef stage1 fill:#ffffff,stroke:#ff0000,stroke-width:3px;
    classDef stage2 fill:#ffffff,stroke:#00ff00,stroke-width:3px;
    classDef stage3 fill:#ffffff,stroke:#0000ff,stroke-width:3px;
    classDef default fill:#ffffff,stroke:#000000,stroke-width:3px;

    input["Input (Real time image)"]:::stage1
    preprocessing["Pre-processing"]:::stage2
    seg["a. Semantic Segmentation<br>using Swin Transformer V2"]:::stage2
    fd["b. Face Detection<br>using MTCNN"]:::stage2
    fc["Face Classification<br>using ShuffleNet"]:::stage2
    od["c. Object Detection (Smartphone)"]:::stage2
    postprocessing["After Processing"]:::stage3
    hand_detect["a. Detect hands in camera<br>(real-time)"]:::stage3
    cheat_item["b. Detect cheating item<br>(real-time)"]:::stage3
    id_detect["c. Detect people ID by face<br>(5 min)"]:::stage3
    problem["Problem Detected"]:::default   
    no_problem["No Problem Detected"]:::default

    input --> preprocessing
    preprocessing --> seg
    preprocessing --> fd
    preprocessing --> od
    fd --> fc
    seg --> postprocessing
    fc --> postprocessing
    od --> postprocessing
    postprocessing --> hand_detect
    postprocessing --> cheat_item
    postprocessing --> id_detect
    hand_detect -->|problem| problem
    cheat_item -->|problem| problem
    id_detect -->|problem| problem
    hand_detect -->|no problem| no_problem
    cheat_item -->|no problem| no_problem
    id_detect -->|no problem| no_problem

'''

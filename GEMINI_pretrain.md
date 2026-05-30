Specs:
This project is run using the submit_and_tail.sh script on the Berkeley Savio compute cluster. NVIDIA 1080 Ti GPUs are used, and each node has 4 GPUs.

Overview:
The script takes in a directory containing files processed_mouse_i.parquet for the ith chunk of data. 
Each data chunk contains around 4000-5000 bulk RNA mouse samples, and each sample containins the expression values for around 16.5k genes (only the genes that have human orthologs).
The expression values are binned, and the BERT-style transformer model learns to predict which bins each masked gene belongs to.

Behavior:
For any bug reports, before proposing changes, first think deeply and simply share your thoughts on why the bug may have occured.
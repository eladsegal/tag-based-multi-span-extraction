set REMOTE=bert-pytorch-vm

set COPY_PATH=project-NLP-AML\debug_wrapper.py
set PATH_TO=./project-NLP-AML/

set COPY_PATH1=project-NLP-AML\src\
set PATH_TO1=./project-NLP-AML/

call gcloud compute scp %COPY_PATH% %REMOTE%:%PATH_TO% --recurse --scp-flag="-p"

REM call gcloud compute scp %COPY_PATH1% %REMOTE%:%PATH_TO1% --recurse --scp-flag="-p"

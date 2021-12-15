
<!---

    Copyright (c) 2021 Robert Bosch GmbH

-->


# CLIN-X
This repository holds the companion code for the system reported in the paper:

"CLIN-X: pre-trained language models and a study on cross-task transfer for concept extraction in the clinical domain" by Lukas Lange, Heike Adel, Jannik Strötgen and Dietrich Klakow.

The paper will be published soon. The code allows the users to reproduce and extend the results reported in the paper. 
Please cite the above paper when reporting, reproducing or extending the results.

    @inproceedings{lange-etal-2021-clin-x,
          author    = {Lukas Lange and
                       Heike Adel and
                       Jannik Str{\"{o}}tgen and
                       Dietrich Klakow},
          title     = {"CLIN-X: pre-trained language models and a study on cross-task transfer for concept extraction in the clinical domain},
          year={2021},
    }

In case of questions, please contact the authors as listed on the paper.

## Purpose of the project
This software is a research prototype, solely developed for and published as part of the publication cited above. It will neither be maintained nor monitored in any way.

    
## The CLIN-X language models
As part of this work, two XLM-R were adapted to the clinical domain 
The models can be found here: 
* **CLIN-X ES**: Spanish clinical XLM-R [(link)](https://huggingface.co/llange/xlm-roberta-large-spanish-clinical)
* **CLIN-X EN**: English clinical XLM-R [(link)](https://huggingface.co/llange/xlm-roberta-large-english-clinical)

The CLIN-X models are open-sourced under the CC-BY 4.0 license. 
See the [LICENSE_models](LICENSE_models) file for details.

    
### Prepare the conda environment
The code requires some python libraries to work: 

    conda create -n clin-x python==3.8.5
    pip install flair==0.8 transformers==4.6.1 torch==1.8.1 scikit-learn==0.23.1 scipy==1.6.3 numpy==1.20.3 nltk tqdm seaborn matplotlib

### Masked-Language-Modeling training
The models were trained using the huggingface MLM script that can be found [here](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py). 
The script was called as follows: 

    python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py  \
    --model_name_or_path xlm-roberta-large  \
    --train_file data/spanisch_clinical_train.txt  \
    --validation_file data/spanisch_clinical_valid.txt  \
    --do_train   --do_eval  \
    --output_dir models/xlm-roberta-large-spanisch-clinical-domain/  \
    --fp16  \
    --per_device_train_batch_size 4 --per_device_eval_batch_size 4  \
    --save_strategy steps --save_steps 10000


    
## Using the CLIN-X model with our propose model architecture (as reported in Table 7)
The following will describe our different scripts to reproduce the results. 
See each of the script files for detailed information on the input arguments. 
    
### Tokenize and split the data

    python tokenize_files.py --input_path path/to/input/files/ --output_path /path/to/bio_files/
    python create_data_splits.py --train_files /path/to/bio_files/ --method random --output_dir /path/to/split_files/

### Train the model (using random data splits)
The following command trains on model on four splits (1,2,3,4) and uses the remaining split (5) for validation. For different split combinations adjust the list of --training_files and the --dev_file arguments accordingly. 

    python train_our_model_architecture.py   \
    --data_path /path/to/split_files/  \
    --train_files random_split_1.txt,random_split_2.txt,random_split_3.txt,random_split_4.txt  \
    --dev_file random_split_5.txt  \
    --model xlm-roberta-large-spanish-clinical  \
    --name model_name --storage_path models
    
### Get ensemble predictions
For all models, get the predictions on the test set as following:

    python get_test_predictions.py --name models/model_name --conll_path /path/to/bio_files/ --out_path predictions/model_name/
    
Then, combine different models into one ensemble. Arguments: Output path + List of model predictions

    python create_ensemble_data.py predictions/ensemble1 predictions/model_name/ predictions/model_name_2/ ...
    


## Using the CLIN-X model (as reported in Table 3)
While we recommand the usage of our model architecture, the CLIN-X models can be used in many other architectures. 
In the paper, we compare to the standard transformer sequnece labeling models as proposed by Devlin et al. 
For this, we provide the `train_standard_model_architecture.py` script

    python train_standard_model_architecture.py  \
    --data_path /path/to/bio_files/  \
    --model xlm-roberta-large-spanish-clinical  \
    --name model_name --storage_path models


    
## License
The CLIN-X code is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Joint-Anonymization-NER, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
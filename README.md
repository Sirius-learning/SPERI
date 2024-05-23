# SPERI:Self periodicity and environment based recurrent imputation for PV production data
This project is related to my proposed algorithm for PV production data imputing with high missing rate of data. We public one PV production dataset in main file names *opened_data.xlsx*.

# Data avaliability
Data support for this paper comes from [the Shandong University Power System Economic Operation Team](https://energymeteo-pseo.sdu.edu.cn/dataset). 
The feature of dataset we used are: TimeStamp, irradiance, realPower.
We declared we haven’t changed the data.

## Index
├── Readme.md                   // help
├── data_handler.py             // process data
├── dataProcess.py 
├── GRUD_model.py               // I handwrote a copy of grud's code based on their paper. Maybe useful
├── main.py 
├── model.py                    // proposed model
├── preprocess.py
├── data                        // Restore some output
│   ├── results_score.txt
│   ├── SDU_GRU_predict_data_20%.csv        // Example output
│   ├── SDU_predict_40%.csv         
├── opened_data.xlsx            // **Our public dataset**


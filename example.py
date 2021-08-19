from GeneralANN import GeneralANN
import pandas as pd

raw_dataset = pd.read_csv("data_RM.csv",
                        na_values=' ', comment='\t',
                        sep=',', skipinitialspace=True)
raw_dataset = raw_dataset.tail(1000)

inp = ['PAR', 'PK1', 'PK2', 'PK3', 'PK4', 'PK5', 'PK6']
out = ['rho'] 

GANN = GeneralANN()
GANN.prepare_data(raw_dataset, inp=inp, out=out, fraction=0.9)
GANN.normalize_data(show_example = False)
GANN.build_and_compile_model([100,50, 20, 1],['relu', 'relu', 'relu'], eta=0.0001)
GANN.train(batch=100, epochs=10)
GANN.test()
GANN.plot_loss()
GANN.plot_scheme()
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_path')
parser.add_argument('--output_path')
args = parser.parse_args()

state_dict = torch.load(args.input_path)
state_dict = state_dict['state_dict']
new_state_dict = 
new_keys = ['backbone.'+key for key in state_dict]
d1 = dict( zip( list(state_dict.keys()), new_keys) )
new_state_dict = {d1[oldK]: value for oldK, value in state_dict.items()}
torch.save(new_state_dict, args.output_path)
from input_pipe import InputStream
import argparse
import logging

# setting up logging config
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Runner:
    '''Responsible for running steps'''
    def __init__(self, step) -> None:
        if step == "data":
            stream = InputStream()
            stream.data_info()



args = argparse.ArgumentParser()
args.add_argument('-r', '--run', help='step need to run \n availabel', choices=['data','preprocess','model','model_stats'],required=True)
args.add_argument('-skip_preprocess', choices=['true','false'])
available_cmd = ['data', 'preprocess', 'model', 'model_stats']
inp_args = vars(args.parse_args())
fn = inp_args.get('run')
try:
    if str(fn) not in available_cmd:
        logging.error('can not find the curresponding run sequence.')
    else:
        runner = Runner(step = fn)
except Exception as e:
    logging.error(f'can not execute the command', e)



from phase2 import task0a, task0b, task1
import argparse

gestures_dir = '../sample/'
k = 20
user_option = 'pca'

window = 3
shift =3
resolution = 3
output_dir = '../outputs/'
vector_model = 'tf_idf'
custom_cost = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create gesture words dictionary.')

    parser.add_argument('--gestures_dir', help='directory of input data', required=False)
    parser.add_argument('--k', type=int, help='k most similar gestures', required=False)
    parser.add_argument('--user_option', help='Type of dimensionality reduction', required=False)

    # optional parameters
    parser.add_argument('--window', type=int, help='window length', required=False)
    parser.add_argument('--shift', type=int, help='shift length', required=False)
    parser.add_argument('--resolution', type=int, help='resolution', required=False)
    parser.add_argument('--output_dir', help='output directory', required=False)
    parser.add_argument('--vector_model', help='vector model', required=False)
    parser.add_argument('--custom_cost', type=bool, help='Custom cost for edit distance', required=False)
    
    args = parser.parse_args()
    
    if args.gestures_dir :
        gestures_dir = args.gestures_dir
    if args.k :
        k = args.k
    if args.user_option : 
        user_option = args.user_option
    if args.window :
        window = args.window
    if args.shift :
        shift = args.shift
    if args.resolution :
        resolution = args.resolution
    if args.output_dir :
        output_dir = args.output_dir
    if args.vector_model :
        vector_model = args.vector_model
    if args.custom_cost :
        custom_cost = args.custom_cost
        
    task0a.call_task0a(gestures_dir, window, shift, resolution)
    task0b.call_task0b(output_dir)
        
    
        
    

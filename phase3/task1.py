from phase2 import task0a, task0b, task1, task3
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create gesture words dictionary.')

    parser.add_argument('--gestures_dir', help='directory of input data', required=True)
    parser.add_argument('--k', type=int, help='k most similar gestures', required=True)
    parser.add_argument('--user_option', help='Type of dimensionality reduction', required=True)

    # optional parameters
    parser.add_argument('--window', type=int, help='window length', required=False)
    parser.add_argument('--shift', type=int, help='shift length', required=False)
    parser.add_argument('--resolution', type=int, help='resolution', required=False)
    parser.add_argument('--output_dir', help='output directory', required=False)
    parser.add_argument('--vector_model', help='vector model', required=False)
    parser.add_argument('--custom_cost', type=bool, help='Custom cost for edit distance', required=False)

    args = parser.parse_args()

    task0a.call_task0a(args.gestures_dir, args.window, args.shift, args.resolution)
    task0b.call_task0b(args.output_dir)
    task1.call_task1(args.output_dir, args.vector_model, args.user_option)  # get for pca trained model
    task3.call_task3(args.vector_model, args.output_dir, args.user_option, 4,
                     "svd", args.custom_cost)  # construct_gesture_gesture_similarity model

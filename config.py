import argparse

def args():
    parser = argparse.ArgumentParser(description='OK')
    parser.add_argument('--train_path', type=str, default='E:\\OneDrive\\Hackathon\\Problem2\\data\\train-problem2.csv')
    parser.add_argument('--test_path', type=str, default='E:\\OneDrive\\Hackathon\\Problem2\\data\\public-problem2.csv')
    parser.add_argument('--data_npy', type=str, default='E:\\OneDrive\\Hackathon\\Problem2\\data\\data.npy')
    parser.add_argument('--target_npy', type=str, default='E:\\OneDrive\\Hackathon\\Problem2\\data\\target.npy')
    parser.add_argument('--pretrained_model_path', type=str, default='vinai/phobert-base')
    parser.add_argument('--rdrsegmenter_path', type=str,  default='E:\\OneDrive\\Hackathon\\Problem2\\vncorenlp')
    parser.add_argument('--max_sequence_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--accumulation_steps', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--checkpoint', type=str, default='models')
    args = parser.parse_args()
    return args
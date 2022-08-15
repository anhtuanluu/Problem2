import argparse

def args():
    parser = argparse.ArgumentParser(description='OK')
    parser.add_argument('--train_path', type=str, default='C:/Users/atuan/Documents/Git/Problem2/data/data_final_problem2.csv')
    parser.add_argument('--test_path', type=str, default='C:/Users/atuan/Documents/Git/Problem2/data/public-problem2.csv')
    parser.add_argument('--data_npy', type=str, default='C:/Users/atuan/Documents/Git/Problem2/data/data_bk.npy')
    parser.add_argument('--target_npy', type=str, default='C:/Users/atuan/Documents/Git/Problem2/data/target.npy')
    parser.add_argument('--pretrained_model_path', type=str, default='vinai/phobert-base')
    parser.add_argument('--rdrsegmenter_path', type=str,  default='C:/Users/atuan/Documents/Git/Problem2/vncorenlp/')
    parser.add_argument('--max_sequence_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulation_steps', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--checkpoint', type=str, default='models')
    parser.add_argument('--head', action='store_true')
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    return args
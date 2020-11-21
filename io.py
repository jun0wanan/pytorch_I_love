
def save_fomat():
    save_file='/home/xiejunlin/jun/dataset/msrvtt-qa/{}/{}_vocab.json'
    with open(args.output_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
         pickle.dump(obj, f)
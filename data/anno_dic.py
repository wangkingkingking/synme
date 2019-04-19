def anno_dic(anno_dic_cache_file, testset_list_file):
    anno_dic = {}
    if os.path.isfile(anno_dic_cache_file):
        with open(anno_dic_cache_file, 'rb') as f:
            anno_dic = pickle.load(f)
    else:
        with open(testset_list_file, 'r') as f:
            lines = f.readlines()
        pairs = [  x.strip().split()  for x in lines]

        image_names = get_image_name_list(testset_list_file)
        for i, imagename in enumerate(image_names):
            anno_dic[imagename] = parse_anno(os.path.join(SYNME_ROOT, pair[1]))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i , len(pairs)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    return anno_dic


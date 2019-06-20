import sys, os, csv

def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')

idx = 0
pred_fpath='./submission/pred{}.csv'.format(idx)
bbox_fpath='./submission/bbox{}.csv'.format(idx)

print('[Info] Prediction file: {}'.format(pred_fpath))
print('[Info] BBox file      : {}'.format(bbox_fpath))

with _open_for_csv(pred_fpath) as file:
    csv_reader = csv.reader(file, delimiter=',', )

    for row in enumerate(csv_reader):
        print(row)

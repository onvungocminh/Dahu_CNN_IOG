
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return 'datasets/'  # folder that contains VOCdevkit/.

        elif database == 'sbd':
            return 'datasets/sbd/'  # folder with img/, inst/, cls/, etc.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        return '/path/to/models/resnet101-5d3b4d8f.pth'    
        #'resnet101-5d3b4d8f.pth' #resnet50-19c8e357.pth'

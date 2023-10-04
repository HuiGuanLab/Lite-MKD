
import logging
import os
import time

class logs(object):
    '''
日志类，run的时候初始化这个类
每次一个类，对应这次运行的时间和该实验的描述
使用这个类会记录实验的参数，准确率和loss
并且这个类会检查参数文件夹并在对应参数文件夹中创建复制log
    '''
    def __init__(self,args) : #需要传入参数
        self.logger = logging.getLogger("strm_logger")
        self.formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')  #日志格式   时间 日志级别名称 日志信息
        # print(args.checkpoint_dir)
        # self.logfile=self.get_assist_log(args.checkpoint_dir)
        self.setup_logger(args.mode)
        
        #这里认为loss应该是warning，训练准确率是 info  ，测试准确率和参数应该是error
        #loss输入的应该是字典

    def setup_logger(self, mode , level = logging.INFO):  
        '''#  创建logger，在代码每跑一次就要创建一个文件，文件名应该是该次运行的时间和实验的描述'''
        
        #正常来说，在运行一个实验的情况下，在初始化类的时候会将logger创建，即每一个实验会有一个logger
        current_time=time.strftime('%Y%m%d%H%M',time.localtime(time.time() ))
        log_name='./log/'+current_time+mode+'.log'
        filehandler = logging.FileHandler(log_name)   #创建日志
        filehandler.setFormatter(self.formatter)   #选择输出格式
        self.logger.setLevel(level)
        self.logger.addHandler(filehandler)
    
        return 
    
    
    '''
    放在这个类里面没有办法运行
    '''
    # def get_assist_log(self,checkpoint_dir):
    #     """
    #     在权重文件夹里面创建一个日志文件
    #     """
    #     logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    #     if os.path.isfile(logfile_path):
    #         logfile = open(logfile_path, "a", buffering=1)
    #     else:
    #         logfile = open(logfile_path, "w", buffering=1)

    #     return logfile
    
    
    
    
    def info(self,message):
        self.logger.info(message)
        
    def warning(self,message):
        self.logger.warning(message)
        
    def error(self,message):
        self.logger.error(message)
 
    # def print_and_log(self, message):
    #     print(message, flush=True)
    #     self.logfile.write(message)  #txt写入
    
    def log_args(self,message):
        '''
        打印和记录实验参数
        message应当接收一个字典，字典里面存放的是实验的参数
        '''
        print('Options:'+'\n')
        for key,value in message.items():
            args_str=key+" is:   "+str(value)+'\n'
            # self.print_and_log(args_str)
            self.error(args_str)
    
    def log_loss(self,message):
        ''' 这里loss有多个，所以也传入一个字典'''
        for key,value in message.items():
            args_str=key+" is :   "+value+'\n'
            print(args_str, flush=True)
            self.logfile.write(args_str)
            self.error(args_str)
            
            


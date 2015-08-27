import subprocess
import os


class MatlabExecuter(object):
    def __init__(self):
        self.exe_str = '-nodisplay -nosplash -nodesktop -r "{};exit;"'
        self._dir = '~'
        self._dir_temp = os.getcwd()

    def _run(self, command):
        os.chdir(self._dir)
        p = subprocess.Popen(['matlab', self.exe_str.format(command)])
        
        return p

    def run_script(self, script):
        return self._run('run({})'.format(script))

    def run_function(self, function):
        return self._run(function)

    def cd(self, path):
        self._dir = path if path[0] == '/' else self._dir + path

    def addpath(self, path):
        self.run_function('addpath(\'{}\')'.format(path))
import subprocess as sp
import os 



path = os.getcwd()

#sp.Popen([str(path), "ls"], cwd = str(path) )


#sp.run(["ux.txt"], cwd = str(path))
#exec(open('MaillageWithStokes.edp').read())

os.system('FreeFem++ MaillageStokesAda.edp')

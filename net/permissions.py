import os, sys, stat
def grant_permissions(filename):
    os.chmod(filename, stat.S_IRWXU | stat.S_IRWXG)
    
def grant_all(path):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            try:
                grant_permissions(os.path.join(root, d))
            except Exception:
                pass
        for f in files:
            try:
                grant_permissions(os.path.join(root, f))
            except Exception:
                pass
	    
path = os.path.abspath("")    
grant_all(path)
print("Set permission 770 for all children of "+path) 

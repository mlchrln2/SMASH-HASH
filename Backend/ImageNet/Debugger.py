'''This module is intended to help debug memory leaks'''
def memory_manager(gc):
	for obj in gc.get_objects():
	    try:
	        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
	            print(type(obj), obj.size())
	    except:
	        pass
 
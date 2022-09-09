import os
from torchmeta import modules

# automatically fix torchmeta and pytorch1.8.1 incompatibility, as discussed
# here: https://github.com/tristandeleu/pytorch-meta/issues/138

convpath = os.path.join(os.path.split(modules.__file__)[0], 'conv.py')
tmppath = os.path.join(os.path.split(modules.__file__)[0], 'tmp.py')
replaced = 'return F.conv2d(F.pad(input, expanded_padding, mode=\'circular\'),'
replacement = 'return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),'

with open(os.path.join(os.path.split(modules.__file__)[0], 'conv.py'), 'r') as orig_file:
    with open(tmppath, 'w') as tmp_file:
        for line in orig_file.readlines():
            if replaced in line:
                line = line.replace(replaced, replacement)
            tmp_file.writelines(line)
    os.remove(convpath)
    os.rename(tmppath, convpath)

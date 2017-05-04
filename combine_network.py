import sys

sys.path.append('/alter/environment/caffe/python/')

import caffe

less_prototxt = '/alter/Codes/getjpgs/big-source/age_gender.py'
less_caffemodel = '/alter/Codes/getjpgs/big-source/_iter_100000.caffemodel'

more_prototxt = "/alter/Codes/getjpgs/big-source/all.prototxt.py"
more_caffemodel = "/alter/Codes/getjpgs/big-source/MCNN2.1.caffemodel"

more_output = '/alter/Codes/getjpgs/big-source/more'
less_output = '/alter/Codes/getjpgs/big-source/less'

save_caffemodel = '/alter/Codes/getjpgs/big-source/result.caffemodel'

#Load the original network and extract the fully connected layers' parameters.
more_net = caffe.Net(more_prototxt,more_caffemodel,caffe.TEST)
count = 0
more_layer_name = []
with open(more_output, 'w') as outf:
    for layer_name, param in more_net.params.iteritems():
        #print 'more_layer_name: ', layer_name, str(param[0].data.shape), str(param[1].data.shape)
        more_layer_name.append(layer_name)
        outf.write(layer_name + '\n')

less_net = caffe.Net(less_prototxt, less_caffemodel, caffe.TEST)

less_layer_name = []
with open(less_output, 'w') as outf:
    for layer_name, param in less_net.params.iteritems():
        #print 'less_layer_name: ', layer_name, str(param[0].data.shape), str(param[1].data.shape)
        less_layer_name.append(layer_name)
        outf.write(layer_name + '\n')

more_layers = []
#origin_layers = []
for layer_name in more_layer_name:
    if layer_name in less_layer_name:
        #origin_layers.append(layer_name)
        continue
    more_layers.append(layer_name)

for layer in more_layers:
    print 'add layer', layer

more_params = {}

more_params = {layer: (more_net.params[layer][0].data, more_net.params[layer][1].data) for layer in more_layers}

target_net = caffe.Net(more_prototxt, less_caffemodel, caffe.TEST)


target_params = {layer: (target_net.params[layer][0].data, target_net.params[layer][1].data) for layer in more_layers}

for layer in more_layers:
    target_params[layer][1][...] = more_params[layer][1]
    target_params[layer][0].flat = more_params[layer][0].flat


target_net.save(save_caffemodel)
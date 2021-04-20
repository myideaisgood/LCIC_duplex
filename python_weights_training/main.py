from config import parse_args
from network import Network    
    
args = parse_args()
my_net = Network(args)
my_net.build()
my_net.train()
my_net.print_all_weights()
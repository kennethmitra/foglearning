# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
from average import average_weights

class Edge():

    def __init__(self, id, cids, shared_layers):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.cids = cids
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.shared_state_dict = shared_layers.state_dict()
        self.clock = []
        self.communication_count = 0

    def refresh_edgeserver(self):
        """
        Clear receiver buffer, id_registration, and sample registration
        :return: None
        """
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        """
        Sign up client for a round of training?
        :param client:
        :return: None
        """
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.train_ds)
        return None

    def receive_from_client(self, client_id, cshared_state_dict):
        """
        Receive model parameters from client
        :param client_id:
        :param cshared_state_dict:
        :return:
        """
        self.receiver_buffer[client_id] = cshared_state_dict
        self.communication_count += 1
        return None

    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.shared_state_dict = average_weights(w = received_dict,
                                                 s_num= sample_num)

    def send_to_client(self, client):
        """
        Send model params from edge to client
        :param client:
        :return:
        """
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud):
        """
        Send model params from edge to cloud server
        :param cloud:
        :return:
        """
        cloud.receive_from_edge(edge_id=self.id,
                                eshared_state_dict= copy.deepcopy(
                                    self.shared_state_dict))
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        """
        Receive model params from cloud server
        :param shared_state_dict:
        :return:
        """
        self.shared_state_dict = shared_state_dict
        return None


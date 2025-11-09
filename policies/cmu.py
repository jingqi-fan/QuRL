class MaxWeightCMuPolicy:
    def __init__(self):
        pass

    def test_forward(self, step, batch_queue, batch_time, repeated_queue, repeated_network, repeated_mu, repeated_h):
        print(f'------------------------cmu------------------------------')
        # print(f'repeated_queue: {repeated_queue}')
        # print(f'repeated_network: {repeated_network}')
        # print(f'repeated_mu: {repeated_mu}')
        # print(f'repeated_h: {repeated_h}')
        # print(f'return {repeated_h * repeated_mu}')
        return repeated_h * repeated_mu


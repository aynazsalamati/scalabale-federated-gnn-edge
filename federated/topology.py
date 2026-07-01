class TopologyManager:
    def __init__(self, num_clients=None):
        self.num_clients = num_clients

    def get_client_weight(self, client_id):
        connectivity = 0.75 + 0.25 * (((client_id * 37) % 10) / 9.0)
        bandwidth = 0.70 + 0.30 * (((client_id * 17 + 3) % 10) / 9.0)
        reliability = 0.80 + 0.20 * (((client_id * 29 + 5) % 10) / 9.0)
        position_score = 0.85 + 0.15 * (((client_id * 11 + 2) % 10) / 9.0)
        score = 0.30 * connectivity + 0.30 * bandwidth + 0.25 * reliability + 0.15 * position_score
        return max(score, 0.1)

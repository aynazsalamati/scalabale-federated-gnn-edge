import copy


class FederatedServer:
    def __init__(self, global_model, device="cpu"):
        self.global_model = global_model.to(device)
        self.device = device

    def aggregate(self, client_updates):
        if not client_updates:
            return self.global_model.state_dict()

        global_weights = copy.deepcopy(client_updates[0][0])
        total_weight = float(client_updates[0][1])

        for key in global_weights:
            global_weights[key] = global_weights[key].float() * total_weight

        for client_weights, importance in client_updates[1:]:
            importance = float(importance)
            total_weight += importance
            for key in global_weights:
                global_weights[key] += client_weights[key].float() * importance

        for key in global_weights:
            global_weights[key] = global_weights[key] / max(total_weight, 1e-12)

        self.global_model.load_state_dict(global_weights)
        return {k: v.detach().cpu().clone() for k, v in global_weights.items()}

class MetricsTracker:
    def __init__(self):
        self.round_losses = []
        self.round_accuracies = []
        self.participation_rates = []
        self.communication_costs = []
        self.client_losses = {}

    def log_round(self, loss, accuracy, participation_rate, communication_cost):
        self.round_losses.append(loss)
        self.round_accuracies.append(accuracy)
        self.participation_rates.append(participation_rate)
        self.communication_costs.append(communication_cost)

    def log_client_loss(self, client_id, loss):
        self.client_losses.setdefault(client_id, []).append(loss)

    def print_summary(self):
        print("\n===== TRAINING SUMMARY =====")
        print(f"Final Accuracy: {self.round_accuracies[-1]:.4f}")
        print(f"Average Participation: {sum(self.participation_rates) / len(self.participation_rates):.2f}")
        print(f"Total Communication Cost: {sum(self.communication_costs)}")

    def get_losses(self):
        return self.round_losses

    def get_accuracies(self):
        return self.round_accuracies

    def get_participation_rates(self):
        return self.participation_rates

    def get_communication_costs(self):
        return self.communication_costs

    def get_client_losses(self):
        return self.client_losses

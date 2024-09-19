from locust import HttpUser, task, between

class PredictUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        self.client.post(
            "/predict",
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            json={
                "url": "https://conx.readthedocs.io/en/latest/_images/MNIST_6_0.png"
            }
        )
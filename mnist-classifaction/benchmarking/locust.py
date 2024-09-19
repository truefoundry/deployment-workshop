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
                "url": "https://github.com/truefoundry/deployment-workshop/blob/main/mnist-classifaction/deploy_model/sample_images/1.jpg?raw=true"
            }
        )
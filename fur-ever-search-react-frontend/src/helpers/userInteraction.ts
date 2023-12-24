import axios from "axios";

// Define a helper function for user interactions
export async function sendUserInteraction(
  uid: string,
  breed: string,
  interactionType: string
) {
  // Make the Axios request
  axios
    .post("http://127.0.0.1:5001/user-interactions", {
      userInteraction: {
        uid: uid,
        breed: breed,
        interaction_type: interactionType,
      },
    })
    .then((response) => {
      const responseData = response?.data;
      console.log("User interactions:", responseData);
    })
    .catch((error) => {
      console.error("Error fetching data:", error);
    });
}

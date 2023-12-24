import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { Dropdown } from "react-bootstrap";
import { v4 } from 'uuid';

import './UserInfo.css';
import { MainNavigation } from '../../components/MainNavigation';
import axios from 'axios';

export interface ILandings {
  landings: any
}

const UserInfo = (props: any) => {
  const [userInfo, setUserInfo] = useState({
    user_id: v4(),
    firstname: '',
    lastname: '',
    gender: '',
    marital_status: '',
    zipcode: '',
    home_type: '',
    children: '',
    family_size: '',
    age: '',
    user_type: 'Real'
  });

  const handleInputChange = (e: any) => {
    const { name, value } = e.target;
    setUserInfo({
      ...userInfo,
      [name]: value,
    });
  };

  const handleSubmit = (e: any) => {
    e.preventDefault();
    console.log('all values', userInfo)
    axios.post(`http://127.0.0.1:5001/breeds`, {
      userInfo
    })
      .then(response => {
        const responseData = response?.data

        console.log('responseData in user info page - jashp', responseData)

        // Iterate through recommendations and fetch breed info for each recommendation
        const fetchRecommendedBreedsInfo = async () => {
          const breedInfoPromises = responseData?.recommendations.map((recommendation: string) =>
            axios.get(`http://127.0.0.1:5000/breeds/${recommendation}`)
          );

          try {
            const breedInfoResponses = await Promise.all(breedInfoPromises);
            const recommendedBreedsData = breedInfoResponses.map(response => {
              const responseBreedData = response.data;
              const breedInfo = responseBreedData?.breed_info[0]
              const images = JSON.parse(breedInfo.images.replace(/'/g, "\""))
              const breedName = breedInfo['\u00ef\u00bb\u00bfbreed']
              return {
                imageUrls: images,
                name: breedName
              }
            });
            console.log('recommendedBreedsData - jashp', recommendedBreedsData)
            const sessionData = {
              userInfo,
              recommendedBreedsData
            }
            const jsonString = JSON.stringify(sessionData);

            sessionStorage.setItem('mySessionData', jsonString);
            console.log('session storage data added for the user')
            console.log('moving to home page with user recommendations')
            window.location.href = '/fur-ever-search/'
            e.preventDefault();
          } catch (error) {
            console.error('Error fetching recommended breed info:', error);
          }
        };

        fetchRecommendedBreedsInfo();
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
  };

  const cartItemCount = useSelector((state: ILandings) => state);
  const backgroundImage = !cartItemCount.landings.isLightTheme ?
    `url('http://localhost:3000/statics/white_home_background.jpg')`
    : `url('http://localhost:3000/statics/black_home_background.jpg')`

  return (
    <div className='App'>
      <MainNavigation cartItemNumber={cartItemCount.landings} />
      <h1 className='userInfoHeading'>User Info</h1>
      <div className="userInfoPageContainer">
        <form onSubmit={handleSubmit}>
          <label>
            First Name:
            <input
              type="text"
              name="firstname"
              value={userInfo.firstname}
              onChange={handleInputChange}
            />
          </label>
          <label>
            Last Name:
            <input
              type="text"
              name="lastname"
              value={userInfo.lastname}
              onChange={handleInputChange}
            />
          </label>
          <br />
          <label>
            Gender:
            <input
              type="radio"
              name="gender"
              value="M"
              onChange={handleInputChange}
            />
            M
            <input
              type="radio"
              name="gender"
              value="F"
              onChange={handleInputChange}
            />
            F
          </label>
          <br />
          <label>
            Zip Code:
            <input
              type="text"
              name="zipcode"
              value={userInfo.zipcode}
              onChange={handleInputChange}
            />
          </label>
          <br />
          <label>
            Marital Status:
            <input
              type="radio"
              name="marital_status"
              value="Single"
              onChange={handleInputChange}
            />
            Single
            <input
              type="radio"
              name="marital_status"
              value="Married"
              onChange={handleInputChange}
            />
            Married
          </label>
          <br />
          <label>
            Family Size:
            <select name="family_size" value={userInfo.family_size} onChange={handleInputChange}>
              <option value="">Select</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
              <option value="5">5+</option>
            </select>
          </label>
          <label>
            Home Type:
            <select name="home_type" value={userInfo.home_type} onChange={handleInputChange}>
              <option value="">Select</option>
              <option value="Apartment">Apartment</option>
              <option value="House">House</option>
              <option value="MobileHome">MobileHome</option>
            </select>
          </label>
          <label>
            Age:
            <input
              type="text"
              name="age"
              value={userInfo.age}
              onChange={handleInputChange}
            />
          </label>
          <br />
          <label>
            Children:
            <input
              type="radio"
              name="children"
              value="True"
              onChange={handleInputChange}
            />
            Yes
            <input
              type="radio"
              name="children"
              value="False"
              onChange={handleInputChange}
            />
            No
          </label>
          <br />
          <button type="submit">Next</button>
          <br />
          <br />
          <br />
          <br />
        </form>
      </div>
    </div>
  );
}

export default UserInfo;

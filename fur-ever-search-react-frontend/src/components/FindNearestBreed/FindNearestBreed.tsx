import React, { useState } from "react";
import "./FindNearestBreed.css"; // Import the CSS file for styling
import axios from "axios";
import { usStateCodes } from './helper'
import { sendUserInteraction } from "../../helpers/userInteraction";

export const FindNearestBreed = ({ breedName }: any) => {
  const [zipcode, setZipcode] = useState("");
  const [stateCode, setStateCode] = useState("");
  const [miles, setMiles] = useState("100");
  const [searchType, setSearchType] = useState("1");

  const handleLookup = (zipcode: string) => {
    // Make a request to the Zippopotam.us API
    axios.get(`https://api.zippopotam.us/us/${zipcode}`)
      .then(response => {
        // Extract the state code from the API response
        const state = response.data.places[0].state;
        setStateCode(usStateCodes[state]?.toLowerCase());
      })
      .catch(error => {
        console.error('Error fetching data:', error);
        setStateCode('Error');
      });
  };

  const handleSearch = () => {
    const storedData = sessionStorage.getItem('mySessionData') || '';
    const retrievedSessionData = storedData && JSON.parse(storedData);
    const uid = retrievedSessionData?.userInfo?.user_id;
    if(uid) {
      sendUserInteraction(uid, breedName, 'Nearbydogs')
    }
    if (searchType == "1") {
      const formattedBreedName = breedName.toLowerCase().replace(/\s+/g, '-');
      const url = `https://marketplace.akc.org/puppies/${formattedBreedName}?location=${zipcode}&page=1&radius=${miles}`;
      window.open(url, "_blank");
    } else {
      const url = `https://www.petfinder.com/search/dogs-for-adoption/us/${stateCode}/${zipcode}/?breed%5B0%5D=${breedName}&distance=${miles}`;
      window.open(url, "_blank");
    }
  };

  return (
    <div className="find-nearest-breed">
      <h2 className="section-name">Search for Nearest {breedName}</h2>
      <br />
      <div className="inputs-container">
        <select
          className="radius-dropdown"
          value={searchType}
          onChange={(e) => setSearchType(e.target.value)}
        >
          <option value="1">AKC Breeder</option>
          <option value="2">Shelter</option>
        </select>
      </div>
      <div className="inputs-container">
        <input
          type="text"
          placeholder="Enter Zipcode"
          className="zipcode-input"
          value={zipcode}
          onChange={(e) => {
            setZipcode(e.target.value)
            handleLookup(e.target.value)
          }}
        />
        <select
          className="radius-dropdown"
          value={miles}
          onChange={(e) => setMiles(e.target.value)}
        >
          <option value="10">10 miles</option>
          <option value="25">25 miles</option>
          <option value="50">50 miles</option>
          <option value="100">100 miles</option>
        </select>
      </div>
      <br />
      <button className="find-button" onClick={handleSearch}>
        Search
      </button>
      <br/>
    </div>
  );
};
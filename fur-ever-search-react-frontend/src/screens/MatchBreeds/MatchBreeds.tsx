import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { Dropdown } from "react-bootstrap";

import './MatchBreeds.css';
import { MainNavigation } from '../../components/MainNavigation';

export interface ILandings {
  landings: any;
}

const MatchBreeds = (props: any) => {
  const attributesList = [
    {"id": 0, "name": "Affectionate With Family"},
    {"id": 1, "name": "Good With Young Children"},
    {"id": 2, "name": "Good With Other Dogs"},
    {"id": 3, "name": "Shedding Level"},
    {"id": 4, "name": "Coat Grooming Frequency"},
    {"id": 5, "name": "Drooling Level"},
    {"id": 6, "name": "Openness To Strangers"},
    {"id": 7, "name": "Playfulness Level"},
    {"id": 8, "name": "Watchdog/Protective Nature"},
    {"id": 9, "name": "Adaptability Level"},
    {"id": 10, "name": "Trainability Level"},
    {"id": 11, "name": "Energy Level"},
    {"id": 12, "name": "Barking Level"},
    {"id": 13, "name": "Mental Stimulation Needs"}
  ];

  const [sliderValues, setSliderValues] = useState(Array(attributesList.length).fill('low'));

  const handleSliderChange = (index: number, value: string) => {
    const newSliderValues = [...sliderValues];
    newSliderValues[index] = value;
    setSliderValues(newSliderValues);
  };

  const handleSubmit = () => {
    // Handle form submission here
    console.log("Slider Values:", sliderValues);
  };

  const cartItemCount = useSelector((state: ILandings) => state);
  const numberOfColumns = 3;
  const attributesPerColumn = Math.ceil(attributesList.length / numberOfColumns);

  return (
    <div className='App'>
      <MainNavigation cartItemNumber={cartItemCount.landings} />
      <h1 className='userInfoHeading'>Self Search</h1>
      <div className="slider-container">
        <div className="slider-columns">
          {[...Array(numberOfColumns)].map((_, columnIndex) => (
            <div className="slider-column" key={columnIndex}>
              {attributesList.slice(columnIndex * attributesPerColumn, (columnIndex + 1) * attributesPerColumn).map((attribute, index) => (
                <div className="slider-wrapper" key={attribute.id}>
                  <label>{attribute.name}</label>
                  <input
                    type="range"
                    min="low"
                    max="high"
                    value={sliderValues[attribute.id]}
                    onChange={(e) => handleSliderChange(attribute.id, e.target.value)}
                  />
                </div>
              ))}
            </div>
          ))}
        </div>
        <button onClick={handleSubmit}>Find</button>
      </div>
    </div>
  );
};

export default MatchBreeds;

import React from 'react';
import { FaCartArrowDown, FaHandHoldingHeart, FaHome, FaUserAlt, FaStream, } from 'react-icons/fa';
import { IoIosLogOut } from 'react-icons/io';
import { useDispatch } from 'react-redux'
import { NavLink } from 'react-router-dom';
import { SiDatadog } from "react-icons/si";


import './MainNavigation.css';

export interface ILandings {
  landings: any
}

function MainNavigation(props: any) {

  const storedData = sessionStorage.getItem('mySessionData') || '';
  const retrievedSessionData = storedData && JSON.parse(storedData);
  const firstname = retrievedSessionData?.userInfo?.firstname
  const lastname = retrievedSessionData?.userInfo?.lastname

  const handleLogout = () => {
    const userConfirmed = window.confirm('Are you sure you want to Logout?');
    if (userConfirmed) {
      console.log('User confirmed. Performing the action...');
      sessionStorage.removeItem('mySessionData');
      window.location.reload();
    } else {
      console.log('User cancelled the action.');
      window.location.reload();
    }
  }

  return (
    <header className='main-navigation'>
      <nav>
        <ul className='home_icon'>
          <li>
            <NavLink to='/fur-ever-search/' title='Home'><FaHome /></NavLink>
          </li>
        </ul>
        <ul className='header_title'>
          <li className='fur-ever-searcher'>
            Fur-ever Search <SiDatadog />
          </li>
        </ul>
        <ul className='my_content'>
          {/* <li>
            <NavLink to='/fur-ever-search/cart' title='Cart'><FaCartArrowDown /> ({props.cartItemNumber.cartSum || 0})</NavLink>
          </li> */}
          {/* <li>
            <NavLink to='/fur-ever-search/liked' title='Liked'><FaHandHoldingHeart /> ({props.cartItemNumber.likedSum || 0})</NavLink>
          </li> */}
          <li>
            <NavLink to='/user-info' title='User Recommendation'>{firstname && lastname && `${firstname}, ${lastname}     `}  <FaUserAlt /></NavLink>
          </li>
          {retrievedSessionData && <li>
            <NavLink to='/user-info' title='Logout' onClick={handleLogout}><b style={{ fontSize: "24px" }}><IoIosLogOut /></b></NavLink>
          </li>}
          {/* <li>
            <NavLink to='/fur-ever-search/profile' title='Profile'><FaUserAlt /></NavLink>
          </li> */}
        </ul>
        {/* <label className='switch'>
          <input type='checkbox' title='Mode Toggler' onChange={() => dispatch({type: 'MODE_TOGGLE'})}/>
          <span className='slider round'></span>
        </label> */}
      </nav>
    </header>
  )
}

export default MainNavigation;

<template>
    <nav class="navbar">
        <div class="title"> <RouterLink to="/overview">價格追蹤小幫手</RouterLink></div>
        <!-- 新增漢堡選單按鈕 -->
        <div class="hamburger" @click="toggleMenu">
            &#9776;
        </div>
        <ul class="options">
            <li><RouterLink to="/overview">物價概覽</RouterLink></li>
            <li><RouterLink to="/trending">物價趨勢</RouterLink></li>
            <li><RouterLink to="/news">相關新聞</RouterLink></li>
            <li v-if="!isLoggedIn"><RouterLink to="/login">登入</RouterLink></li>
            <li v-else @click="logout">Hi, {{getUserName}}! 登出</li>
        </ul>
    </nav>
</template>

<script>
import { useAuthStore } from '@/stores/auth';

export default {
    name: 'NavBar',
    data() {
        return {
            isMenuOpen: false // 控制漢堡選單的開關狀態
        };
    },
    computed: {
        isLoggedIn(){
            const userStore = useAuthStore();
            return userStore.isLoggedIn;
        },
        getUserName(){
            const userStore = useAuthStore();
            return userStore.getUserName;
        }
    },
    methods: {
        toggleMenu() {
            this.isMenuOpen = !this.isMenuOpen; // 切換選單顯示狀態
        },
        logout(){
            const userStore = useAuthStore();
            userStore.logout();
        }
    }
};
</script>

<style scoped>
.navbar {
    position: sticky;
    display: flex;
    justify-content: space-between;
    background-color: #f3f3f3;
    padding: 1.5em;
    height: 4.5em;
    width: 100%;
    align-items: center;
    box-shadow: 0 0 5px #000000;
    white-space: nowrap; /* 確保不要換行 */
}

.navbar ul {
    list-style: none;
    display: flex;
    justify-content: space-around;
}

.title > a{
    font-size: 1.4em;
    font-weight: bold;
    color: #2c3e50 !important;
}

.navbar li {
    color: #575B5D;
    margin: 0 .5em;
    font-size: 1.2em;
}

.navbar li:hover{
    cursor: pointer;
    font-weight: bold;
}

.navbar a {
    text-decoration: none;
    color: #575B5D;
}

/* 隱藏漢堡選單按鈕，當螢幕大於 768px 時 */
.hamburger {
    display: none;
    font-size: 2em;
    cursor: pointer;
}

/* 當螢幕小於 768px 時，隱藏選項列表，顯示漢堡選單 */
@media (max-width: 768px) {
    .navbar{
        height: auto;
    }
    .navbar ul {
        display: none;
        flex-direction: column;
        position: fixed;
        top: 4.5em;
        left: 0;
        background-color: #f3f3f3;
        width: 100%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }

    .navbar ul.active {
        display: flex; /* 當漢堡選單開啟時顯示選項列表 */
    }

    .hamburger {
        display: block; /* 顯示漢堡選單按鈕 */
    }

    .navbar li {
        width: 100%;
        padding: 1em;
        text-align: center;
        border-bottom: 1px solid #ddd; /* 增加底部分隔線 */
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .navbar li:last-child {
        border-bottom: none; /* 最後一個選項不需要底部分隔線 */
    }
}

</style>
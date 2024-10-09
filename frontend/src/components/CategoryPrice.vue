<template>
    <div class="category-price-wrapper">
        <h2>{{ categoryName }}</h2>
        <div v-if="isLoading">Loading...</div>
        <div v-if="errorMessage" class="error">{{ errorMessage }}</div>
        <table v-if="!isLoading && !errorMessage">
            <thead>
                <tr>
                    <th>商品名稱</th>
                    <th>規格</th>
                    <th>{{latestDataTime}} 最新價格</th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="data in priceData" :key="data.編號">
                    <td>{{ data.產品名稱 }}</td>
                    <td>{{ data.規格 }}</td>
                    <td>{{ latestPrice(data.統計值) }}</td>
                </tr>
            </tbody>
        </table>
    </div>
</template>

<script>
import Categories from '@/constants/categories';

export default {
    props: {
        category: {
            type: String,
            required: true
        },
        priceData: {
            type: Array,
            required: true
        },
        isLoading: {
            type: Boolean,
            required: true
        },
        errorMessage: {
            type: String,
            required: false
        },
    },
    computed: {
        categoryName() {
            return Categories[this.category];
        },
        latestDataTime(){
            let timeTmp = this.priceData[0].時間終點.split('-');
            return timeTmp[0] + '.' + timeTmp[1];
        }
    },
    methods: {
        latestPrice(prices_str) {
            let number = prices_str.split(',').map(Number);
            let i = number.length - 1;
            while (i >= 0 && number[i]==0) {
                i--;
            }
            return i==-1 ? "-" : number[i];
        }
    }
};
</script>

<style scoped>
.error {
    color: red;
}
table {
    width: 100%;
    border-collapse: collapse;
    background-color: white;
    white-space: nowrap; /* 確保文字不會換行 */
    /* text-align: center; */
}
.table-container {
    width: 100%;
    overflow-x: auto; /* 讓表格在寬度不足時可以橫向捲動 */
}
th, td {
    border: 1px solid #ddd;
    text-align: center;
    padding: 0.5rem;
}
th{
    background-color: #355f81;
    color: white;
}
h2{
    margin-bottom: .5em;
    font-size: 1.5em;
    font-weight: bold;
}
.category-price-wrapper{
    background-color: white;
    border-radius: 1em;
    padding: 2em;
    overflow-x: auto; /* 手機上允許橫向捲動 */
    width: 100%;
}
/* 手機響應式樣式 */
@media screen and (max-width: 768px) {
    th, td {
        padding: 0.3rem;
        font-size: 0.9rem; /* 調整字體大小以適應手機 */
    }

    .category-price-wrapper h2 {
        font-size: 1.5rem; /* 調整標題大小 */
    }

    table {
        width: 100%;
        font-size: 0.9rem;
    }
}
</style>
